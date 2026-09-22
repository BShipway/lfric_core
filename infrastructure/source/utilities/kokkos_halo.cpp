//----------------------------------------------------------------------------
// (C) Crown copyright 2026 Met Office. All rights reserved.
// The file LICENCE, distributed with this code, contains details of the terms
// under which the code may be used.
//----------------------------------------------------------------------------
//
// The C++ half of halo_exchange_plan_mod: pack and unpack a halo exchange on
// the card.
//
// What this removes. A halo exchange on a device-resident field currently
// leaves the card entirely. YAXT hands MPI an indexed datatype, OpenMPI's
// datatype convertor walks the owned block on the CPU, and every page it
// touches faults back across the bus. In a six-rank C144 step that gather --
// opal_generic_simple_pack -- is 0.290 s a step a rank, about 12% of the step,
// and it is the one part of the exchange whose cost transfers to Phase 8
// hardware: a gather costs what it costs whether the neighbour is on this card,
// the next card or the next node.
//
// What replaces it. The send positions are read out of YAXT's own exchange map
// (see halo_exchange_plan_mod's header), copied to the card once per routing
// and depth, and a kernel gathers with them into a packed device buffer. One
// blocking copy moves that buffer to a pinned host mirror, MPI sends the mirror
// as a plain contiguous block of one datatype, and the receive side runs the
// same three steps backwards. The owned block never leaves the card.
//
// Why a pinned mirror rather than sending from managed memory directly.
// Managed memory is a legal MPI buffer on this stack, but handing OpenMPI a
// managed pointer puts the pages wherever its progress engine last touched
// them, which is the fault the exchange is being taken off the host to avoid.
// A CudaHostPinnedSpace mirror is host memory with a fixed physical address:
// the device copy into it is a DMA that does not fault, and the MPI send out of
// it is a memcpy from a page the transport can register once. The cost is one
// extra copy of the packed buffer, which is tens of kilobytes, against a walk
// of the whole owned block, which is tens of megabytes.
//
// The plan is created once per routing, depth and Fortran kind, and kept. The
// index lists do not depend on the kind but the buffers do, so the element size
// is fixed at creation and a call that disagrees with it is refused rather than
// reinterpreted.
//
// USE_KOKKOS reaches this file through CXXFLAGS rather than through
// PRE_PROCESS_MACROS, for the reason kokkos_runtime.cpp gives. Without it this
// is an empty translation unit, which is what a Fortran-only build wants.

#ifdef USE_KOKKOS

#include <Kokkos_Core.hpp>

#include <cstddef>
#include <cstdint>

namespace {

// As in kokkos_reduce.cpp: the memory space is named rather than left as
// AnonymousSpace, because nvcc 13.3 in this toolchain has been seen to
// mis-infer an AnonymousSpace handle's address space (playbook lesson 90).
using ExecSpace = Kokkos::DefaultExecutionSpace;
using MemSpace = ExecSpace::memory_space;

// Where the staging mirror lives. On a CUDA build that is pinned host memory,
// which is the whole point of the mirror. On a host build there is no bus to
// cross, the mirror is ordinary host memory, and the two copies either side of
// the exchange are memcpys that the compiler is free to make cheap -- the path
// still runs, and still gives the same answer, which is what makes a host build
// usable as a correctness gate for it.
#ifdef KOKKOS_ENABLE_CUDA
using PinnedSpace = Kokkos::CudaHostPinnedSpace;
#else
using PinnedSpace = Kokkos::HostSpace;
#endif

template <typename T, typename Space>
using Buffer = Kokkos::View<T *, Space>;

template <typename T>
using UnmanagedDevice =
    Kokkos::View<T *, MemSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>>;

template <typename T>
using ConstUnmanagedDevice =
    Kokkos::View<const T *, MemSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>>;

template <typename T>
using UnmanagedPinned =
    Kokkos::View<T *, PinnedSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>>;

// The packed buffers are allocated as doubles and reinterpreted, so that a
// 32-bit exchange gets an eight-byte-aligned base rather than whatever a
// char allocation happened to give it.
using Word = double;

std::size_t words_for(std::size_t count, int element_bytes)
{
  const std::size_t bytes = count * static_cast<std::size_t>(element_bytes);
  return (bytes + sizeof(Word) - 1) / sizeof(Word);
}

struct HaloPlan
{
  // Zero-based positions in the field array, on the card.
  Buffer<int, MemSpace> send_index;
  Buffer<int, MemSpace> recv_index;
  // The packed messages: one copy on the card, one pinned host mirror.
  Buffer<Word, MemSpace> send_device;
  Buffer<Word, MemSpace> recv_device;
  Buffer<Word, PinnedSpace> send_host;
  Buffer<Word, PinnedSpace> recv_host;
  std::size_t n_send = 0;
  std::size_t n_recv = 0;
  int element_bytes = 0;
};

// Copy a one-based Fortran index list to the card as zero-based.
Buffer<int, MemSpace> to_device_zero_based(const int *index, std::size_t n)
{
  Buffer<int, MemSpace> device("lfric_kokkos_halo_index", n);
  auto mirror = Kokkos::create_mirror_view(device);
  for (std::size_t k = 0; k < n; ++k)
  {
    mirror(k) = index[k] - 1;
  }
  Kokkos::deep_copy(device, mirror);
  return device;
}

template <typename T>
void gather_into_mirror(HaloPlan *plan, const T *field, std::size_t n_field)
{
  if (plan->n_send == 0)
  {
    return;
  }
  ConstUnmanagedDevice<T> source(field, n_field);
  UnmanagedDevice<T> packed(reinterpret_cast<T *>(plan->send_device.data()),
                            plan->n_send);
  auto index = plan->send_index;
  Kokkos::parallel_for(
      "lfric_kokkos_halo_gather", Kokkos::RangePolicy<ExecSpace>(0, plan->n_send),
      KOKKOS_LAMBDA(const std::size_t k) { packed(k) = source(index(k)); });
  // One fence, and it is this deep_copy's: a blocking deep_copy waits for the
  // kernel that filled the buffer, so a Kokkos::fence() beside it would be a
  // second whole-device wait for something already waited on.
  UnmanagedPinned<T> mirror(reinterpret_cast<T *>(plan->send_host.data()),
                            plan->n_send);
  Kokkos::deep_copy(mirror, packed);
}

template <typename T>
void scatter_from_mirror(HaloPlan *plan, T *field, std::size_t n_field)
{
  if (plan->n_recv == 0)
  {
    return;
  }
  UnmanagedPinned<T> mirror(reinterpret_cast<T *>(plan->recv_host.data()),
                            plan->n_recv);
  UnmanagedDevice<T> packed(reinterpret_cast<T *>(plan->recv_device.data()),
                            plan->n_recv);
  Kokkos::deep_copy(packed, mirror);
  UnmanagedDevice<T> target(field, n_field);
  auto index = plan->recv_index;
  Kokkos::parallel_for(
      "lfric_kokkos_halo_scatter",
      Kokkos::RangePolicy<ExecSpace>(0, plan->n_recv),
      KOKKOS_LAMBDA(const std::size_t k) { target(index(k)) = packed(k); });
  // The scatter's result is read next by whatever kernel or host code uses the
  // halo, and the host is the more likely of the two, so the wait is taken
  // here rather than left for the reader to remember.
  Kokkos::fence("lfric_kokkos_halo_scatter");
}

}  // namespace

// Every entry point is called only after halo_exchange_plan_mod has checked
// that the Kokkos runtime is running and that the field's storage came from
// the shared allocator, so no call here has to decide whether the card can
// reach the pointer it was given.

extern "C" void *lfric_kokkos_halo_create(const int *send_index,
                                          std::size_t n_send,
                                          const int *recv_index,
                                          std::size_t n_recv,
                                          int element_bytes)
{
  if (element_bytes != 4 && element_bytes != 8)
  {
    return nullptr;
  }
  HaloPlan *plan = new HaloPlan();
  plan->n_send = n_send;
  plan->n_recv = n_recv;
  plan->element_bytes = element_bytes;
  if (n_send > 0)
  {
    plan->send_index = to_device_zero_based(send_index, n_send);
    plan->send_device = Buffer<Word, MemSpace>(
        "lfric_kokkos_halo_send_device", words_for(n_send, element_bytes));
    plan->send_host = Buffer<Word, PinnedSpace>(
        "lfric_kokkos_halo_send_host", words_for(n_send, element_bytes));
  }
  if (n_recv > 0)
  {
    plan->recv_index = to_device_zero_based(recv_index, n_recv);
    plan->recv_device = Buffer<Word, MemSpace>(
        "lfric_kokkos_halo_recv_device", words_for(n_recv, element_bytes));
    plan->recv_host = Buffer<Word, PinnedSpace>(
        "lfric_kokkos_halo_recv_host", words_for(n_recv, element_bytes));
  }
  return static_cast<void *>(plan);
}

extern "C" void lfric_kokkos_halo_destroy(void *handle)
{
  delete static_cast<HaloPlan *>(handle);
}

// The two mirrors, as addresses MPI can be handed. They are stable for the
// life of the plan, so the Fortran side associates its buffer pointers once.
extern "C" void *lfric_kokkos_halo_send_host(void *handle)
{
  HaloPlan *plan = static_cast<HaloPlan *>(handle);
  return (plan == nullptr || plan->n_send == 0)
             ? nullptr
             : static_cast<void *>(plan->send_host.data());
}

extern "C" void *lfric_kokkos_halo_recv_host(void *handle)
{
  HaloPlan *plan = static_cast<HaloPlan *>(handle);
  return (plan == nullptr || plan->n_recv == 0)
             ? nullptr
             : static_cast<void *>(plan->recv_host.data());
}

extern "C" int lfric_kokkos_halo_gather_real32(void *handle, const float *field,
                                               std::size_t n_field)
{
  HaloPlan *plan = static_cast<HaloPlan *>(handle);
  if (plan == nullptr || plan->element_bytes != 4) return 0;
  gather_into_mirror(plan, field, n_field);
  return 1;
}

extern "C" int lfric_kokkos_halo_gather_real64(void *handle,
                                               const double *field,
                                               std::size_t n_field)
{
  HaloPlan *plan = static_cast<HaloPlan *>(handle);
  if (plan == nullptr || plan->element_bytes != 8) return 0;
  gather_into_mirror(plan, field, n_field);
  return 1;
}

extern "C" int lfric_kokkos_halo_scatter_real32(void *handle, float *field,
                                                std::size_t n_field)
{
  HaloPlan *plan = static_cast<HaloPlan *>(handle);
  if (plan == nullptr || plan->element_bytes != 4) return 0;
  scatter_from_mirror(plan, field, n_field);
  return 1;
}

extern "C" int lfric_kokkos_halo_scatter_real64(void *handle, double *field,
                                                std::size_t n_field)
{
  HaloPlan *plan = static_cast<HaloPlan *>(handle);
  if (plan == nullptr || plan->element_bytes != 8) return 0;
  scatter_from_mirror(plan, field, n_field);
  return 1;
}

#endif
