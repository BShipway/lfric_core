//----------------------------------------------------------------------------
// (C) Crown copyright 2026 Met Office. All rights reserved.
// The file LICENCE, distributed with this code, contains details of the terms
// under which the code may be used.
//----------------------------------------------------------------------------
//
// The C++ half of kokkos_memory_mod: field storage taken from
// Kokkos::SharedSpace, so that one address is valid to both the Fortran that
// owns the field and the generated Kokkos region that reads it.
//
// Kokkos::SharedSpace is CudaUVMSpace, HIPManagedSpace or SYCLSharedUSMSpace
// on an accelerator and HostSpace on a host-only build, so this file costs
// nothing measurable in the CPU-only image and needs no second code path when
// a device appears.
//
// USE_KOKKOS reaches this file through CXXFLAGS rather than through
// PRE_PROCESS_MACROS, for the reason kokkos_runtime.cpp gives: the C++ rule in
// compile.mk passes CXXFLAGS and the Fortran preprocessor macros are a separate
// list. Without it this is an empty translation unit, which is what a
// Fortran-only build wants.
//
// The accounting below is not instrumentation that could be dropped. It is the
// only evidence available on a host-only build that field data came from the
// shared allocator at all: SharedSpace aliases HostSpace here, so the model's
// numbers are identical either way and no checksum can tell the two apart.

#ifdef USE_KOKKOS

#include <Kokkos_Core.hpp>

#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <unordered_map>

#if defined(KOKKOS_ENABLE_CUDA)
#include <cuda_runtime.h>
#endif

// Kokkos_Core_fwd.hpp defines has_shared_space true in every configuration that
// resolves the SharedSpace alias, and leaves it false for OpenACC and
// OpenMPTarget builds, where the alias does not exist. Failing here is far
// easier to read than the template error the first use would otherwise raise.
static_assert(Kokkos::has_shared_space,
              "Kokkos was built without a shared memory space");

namespace {

// Field construction is serial -- it happens on the master thread inside the
// model's initialise -- so these need no synchronisation. If that ever stops
// being true, the symptom is a wrong byte count rather than a wrong field.
std::size_t live_bytes = 0;
std::size_t peak_bytes = 0;
std::size_t live_blocks = 0;

// Every pointer the allocator has issued and not yet reclaimed, against the
// index of the Fortran registry entry that owns it. The set of keys exists so
// that the free below can answer "did I allocate this?" per pointer. The
// allocator falls back to returning nullptr when the runtime is not up, so a
// run may legitimately hold a mixture of shared-space arrays and ordinary
// Fortran ALLOCATEd ones, and freeing either of them the other way round is
// memory corruption. No flag on the run as a whole can separate them, because
// the property belongs to each array.
//
// The value is carried so that kokkos_memory_mod can release one block by
// address in constant time rather than by scanning its registry. It is a
// one-based Fortran index, and zero means "issued, but no registry entry owns
// it" -- the state of every block that came from kokkos_shared_allocate
// rather than from kokkos_shared_claim. The map is the only index: a block
// that fell back to Fortran ALLOCATE never appears here, and the Fortran side
// scans for those.
std::unordered_map<void *, std::size_t> issued;

// MEMORY ADVICE: where a shared-space block should live, and who else may
// touch it. A SharedSpace block on a CUDA build is managed (UVM) memory whose
// pages migrate to whichever processor touches them, one fault at a time.
// Phase 7's first measurement on an H100 (2026-09-12, C16_MG, four
// timesteps) saw 2.6 GB migrate each way against a 240 MB field set, with
// 170 thousand GPU page faults, and the device model stepping slower than
// one Fortran core. The advice is the cheapest lever the strategy names
// (docs/strategy/2026-09-03-device-memory.md; phase-7 plan, Task B3):
//
//   LFRIC_KOKKOS_MEM_ADVICE=none    the runtime's default; every page migrates
//                                   on touch. What every run before this knob
//                                   did.
//   LFRIC_KOKKOS_MEM_ADVICE=device  prefer the device: pages are prefetched to
//                                   the card at allocation and stay there, and
//                                   the CPU is declared an accessor, so a host
//                                   read of a field maps the page across the
//                                   bus instead of migrating it back.
//   LFRIC_KOKKOS_MEM_ADVICE=host    the mirror image: pages stay on the host
//                                   and the device is the accessor. The
//                                   control that says whether "advice at all"
//                                   or "the device end" is what moves a figure.
//
// Read once, on the first allocation, and reported on the shared report
// line. Anything but the three names is refused to `none` with a message,
// not silently. On a host-only build the advice compiles to nothing: the
// alias is HostSpace and there is no other processor to prefer. The CUDA
// 13 signatures take a cudaMemLocation; the older int-device forms are
// gone from that toolkit, which is why the struct is built here.
enum class Advice { none, device, host };

Advice advice_mode()
{
  static const Advice mode = [] {
    const char *value = std::getenv("LFRIC_KOKKOS_MEM_ADVICE");
    if (value == nullptr || *value == '\0' || std::strcmp(value, "none") == 0) {
      return Advice::none;
    }
    if (std::strcmp(value, "device") == 0) {
      return Advice::device;
    }
    if (std::strcmp(value, "host") == 0) {
      return Advice::host;
    }
    std::fprintf(stderr,
                 "lfric_kokkos_shared: LFRIC_KOKKOS_MEM_ADVICE=%s is not none, "
                 "device or host; using none\n", value);
    return Advice::none;
  }();
  return mode;
}

const char *advice_name()
{
  switch (advice_mode()) {
    case Advice::device: return "device";
    case Advice::host: return "host";
    default: return "none";
  }
}

// Counted so the report line can say how many blocks were advised and how
// many of those advice calls the runtime refused, which on a card without
// the feature (or a host-only build) is every one of them or none.
std::size_t advised_blocks = 0;
std::size_t advice_failures = 0;

void advise(void *pointer, std::size_t nbytes)
{
#if defined(KOKKOS_ENABLE_CUDA)
  const Advice mode = advice_mode();
  if (mode == Advice::none || pointer == nullptr || nbytes == 0) {
    return;
  }
  int device = 0;
  if (cudaGetDevice(&device) != cudaSuccess) {
    advice_failures += 1;
    return;
  }
  cudaMemLocation on_device{};
  on_device.type = cudaMemLocationTypeDevice;
  on_device.id = device;
  cudaMemLocation on_host{};
  on_host.type = cudaMemLocationTypeHost;
  on_host.id = 0;
  const cudaMemLocation &preferred = (mode == Advice::device) ? on_device : on_host;
  const cudaMemLocation &accessor = (mode == Advice::device) ? on_host : on_device;
  bool ok = true;
  ok = ok && cudaMemAdvise(pointer, nbytes, cudaMemAdviseSetPreferredLocation,
                           preferred) == cudaSuccess;
  ok = ok && cudaMemAdvise(pointer, nbytes, cudaMemAdviseSetAccessedBy,
                           accessor) == cudaSuccess;
  if (ok && mode == Advice::device) {
    // Start the pages where they are wanted rather than faulting them over
    // one at a time on the first region entry. Asynchronous on the default
    // stream; the first region's fence orders it.
    ok = cudaMemPrefetchAsync(pointer, nbytes, on_device, 0, nullptr) == cudaSuccess;
  }
  if (ok) {
    advised_blocks += 1;
  } else {
    advice_failures += 1;
    // Clear a sticky error so the next CUDA call is not blamed for this one.
    (void)cudaGetLastError();
  }
#else
  (void)pointer;
  (void)nbytes;
#endif
}

}  // namespace

// Returns nullptr rather than aborting when the runtime is not up, so that a
// caller outside Kokkos's lifetime gets an answer it can act on. This is the
// defensive posture kokkos_runtime.cpp takes for the same reason: an abort
// during model start-up is far harder to read than a clean fallback.
extern "C" void *lfric_kokkos_shared_allocate(std::size_t nbytes)
{
  if (nbytes == 0) {
    return nullptr;
  }
  if (!Kokkos::is_initialized() || Kokkos::is_finalized()) {
    return nullptr;
  }

  void *pointer = Kokkos::kokkos_malloc<Kokkos::SharedSpace>(nbytes);
  if (pointer != nullptr) {
    advise(pointer, nbytes);
    issued[pointer] = 0;
    live_bytes += nbytes;
    live_blocks += 1;
    if (live_bytes > peak_bytes) {
      peak_bytes = live_bytes;
    }
  }
  return pointer;
}

// Returns 1 if this allocator owned the pointer and has freed it, and 0 if it
// did not, in which case the memory is the caller's to DEALLOCATE. The byte
// count is a parameter because kokkos_free does not need it but the accounting
// does.
extern "C" int lfric_kokkos_shared_free(void *pointer, std::size_t nbytes)
{
  if (pointer == nullptr) {
    return 0;
  }

  const auto found = issued.find(pointer);
  if (found == issued.end()) {
    return 0;
  }

  issued.erase(found);
  Kokkos::kokkos_free<Kokkos::SharedSpace>(pointer);
  live_bytes -= nbytes;
  live_blocks -= 1;
  return 1;
}

// Returns the one-based registry index recorded against a pointer, or zero if
// this allocator did not issue it or no registry entry owns it. A caller
// cannot tell those two apart from the answer, and does not need to: both
// mean "not findable here, look elsewhere".
extern "C" std::size_t lfric_kokkos_shared_index(void *pointer)
{
  if (pointer == nullptr) {
    return 0;
  }

  const auto found = issued.find(pointer);
  if (found == issued.end()) {
    return 0;
  }
  return found->second;
}

// Records the one-based registry index owning a pointer, doing nothing if this
// allocator did not issue it. Called when a block is claimed, and again when a
// release moves the registry's last entry into the hole the released block
// left, which changes that entry's index.
extern "C" void lfric_kokkos_shared_set_index(void *pointer, std::size_t index)
{
  if (pointer == nullptr) {
    return;
  }

  const auto found = issued.find(pointer);
  if (found != issued.end()) {
    found->second = index;
  }
}

extern "C" std::size_t lfric_kokkos_shared_bytes()
{
  return live_bytes;
}

extern "C" std::size_t lfric_kokkos_shared_peak_bytes()
{
  return peak_bytes;
}

// One line, to stderr. Not through log_mod: gungho_model finalises the logger
// before it finalises Kokkos, so no logger exists by the time this is called.
// The prefix through "blocks" is matched by psy-ir-aidev's
// tests/test_field_data_is_shared.sh and read by bin/measure-timestep, so it
// is kept exactly; the advice fields are appended after it.
extern "C" void lfric_kokkos_shared_report()
{
  std::fprintf(stderr,
               "lfric_kokkos_shared: peak %zu bytes, "
               "%zu bytes live in %zu blocks, advice=%s advised=%zu refused=%zu\n",
               peak_bytes, live_bytes, live_blocks,
               advice_name(), advised_blocks, advice_failures);
}

#endif
