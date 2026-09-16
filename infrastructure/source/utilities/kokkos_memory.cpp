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
#include <map>
#include <unordered_map>
#include <vector>

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
// model's initialise -- so these need no synchronisation. The free list below
// is held under the same assumption, and it is the stronger claim of the two:
// a race on a counter costs a wrong byte count, but a race on the free list
// would hand one block to two callers. The assumption is stated here rather
// than defended by a mutex because taking a lock on every field allocation to
// protect a sequence that is serial by construction would be paying for a
// hazard the model does not have; if field construction ever moves off the
// master thread, this file needs a lock before it needs anything else.
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


// THE FREE LIST: why a shared-space block is worth keeping.
//
// A field's data is allocated and released on a schedule the model sets, and
// at C48 the step allocates and releases about nine hundred shared blocks --
// the same sizes, every step, because the field set does not change. Each one
// is a driver call, and on a CUDA build a driver call for managed memory is
// not cheap in the way an ordinary malloc is. Two costs, both measured or
// read rather than assumed:
//
//   * Kokkos brackets the managed allocation and the managed free with a
//     device synchronisation on each side. Confirmed in the image's own
//     Kokkos 4.7.04 (2026-09-16) by reading the fence names compiled into
//     libkokkoscore: "Kokkos::CudaUVMSpace::impl_allocate: Pre UVM
//     Allocation" and "... Post UVM Allocation", and the matching pair
//     "impl_deallocate: Pre UVM Deallocation" and "... Post UVM
//     Deallocation". Kokkos_CudaSpace.cpp itself is not shipped in the
//     image; the strings in the built library are the evidence. Four whole-
//     device fences per block per step, each one draining the card.
//
//   * kokkos_malloc<SharedSpace> is SharedAllocationRecord::allocate_tracked
//     (Kokkos_Core.hpp), so the block carries a SharedAllocationHeader inside
//     the managed allocation, written by the host. A fresh managed page is on
//     no processor yet, so that write is a host page fault before the field
//     has been touched at all.
//
// Keeping the storage removes all of it: no driver call, no fence pair, and
// the header page is already where it was left. Nothing about the *contents*
// is reused -- a re-issued block holds whatever the last owner left, exactly
// as a fresh managed block holds whatever the driver left, and every caller
// of this allocator writes before it reads, as it must today.
//
// The knob is off by default. That is the owner's decision of 2026-09-16 and
// not a property of the mechanism: it stays off until the measurement says
// otherwise, and the report line says which way it resolved on every run.
bool pooling()
{
  static const bool setting = [] {
    const char *value = std::getenv("LFRIC_KOKKOS_SHARED_POOL");
    return value != nullptr && std::strcmp(value, "1") == 0;
  }();
  return setting;
}

// How many bytes of unused field storage the free list may hold, in
// megabytes. A bound rather than none, for the reason the staging pool's
// LFRIC_KOKKOS_STAGING_POOL_MB has one: the key is an exact byte count, so a
// run meeting many distinct sizes would keep a spare of each for ever, and
// device memory is the scarce resource on this card. Past the bound a
// released block goes back to the driver as it did before the pool existed.
// The default is generous against the model's own field set -- C144's peak is
// a little over a gigabyte -- and small against an H100's 94 GB.
std::size_t pool_limit()
{
  static const std::size_t bytes = [] {
    const char *value = std::getenv("LFRIC_KOKKOS_SHARED_POOL_MB");
    const long megabytes = (value == nullptr || *value == '\0')
                               ? 4096 : std::atol(value);
    return megabytes <= 0 ? std::size_t(0)
                          : std::size_t(megabytes) * 1024 * 1024;
  }();
  return bytes;
}

// Blocks that have been released and not yet re-issued, by exact byte count.
// Exact, not best fit: over-serving a small request with a large block would
// make the accounting lie about how much memory the run holds, and the field
// set repeats its sizes every step, so an exact key serves every request a
// best fit would have served. A pooled block is NOT live -- live_bytes,
// live_blocks and the issued map all exclude it -- because "live" answers
// "what does a field still hold?" and the pool's answer to that is nothing.
// What the pool holds is reported separately, as held_peak.
std::map<std::size_t, std::vector<void *>> spares;
std::size_t pool_bytes = 0;      // held unused right now
std::size_t pool_peak = 0;       // the most ever held unused
std::size_t pool_reuses = 0;     // requests served from the free list
std::size_t pool_recycled = 0;   // releases kept rather than given back
std::size_t pool_released = 0;   // pooled blocks handed back to the driver

// Give every pooled block back to the driver. Called from the finalize hook
// below, and so before Kokkos::finalize: a SharedSpace block cannot be
// returned once the runtime has gone, which is the same reason
// kokkos_memory_mod's release_all is a shutdown operation.
void empty_pool()
{
  for (auto &entry : spares) {
    for (void *pointer : entry.second) {
      Kokkos::kokkos_free<Kokkos::SharedSpace>(pointer);
      pool_released += 1;
    }
    entry.second.clear();
  }
  spares.clear();
  pool_bytes = 0;
}

// Registered once, on the first allocation made with the pool on, for the
// reason the generated staging header registers its own: a hook is the only
// place that runs while Kokkos is still up but after every caller has
// finished with it. The Fortran side's release_all runs earlier still and
// releases every claimed block *into* this pool; this hook is what then
// passes the pool on to the driver. Without it the run would end with the
// held blocks never freed, which Kokkos reports as leaked allocations.
void ensure_finalize_hook()
{
  static const bool registered = [] {
    Kokkos::push_finalize_hook([] {
      const std::size_t held = pool_bytes;
      empty_pool();
      if (held > 0) {
        // A line of its own, and after the report rather than inside it:
        // kokkos_shared_report is called from the Fortran finalise, which
        // runs before Kokkos::finalize and so before this hook. released= on
        // the report line is therefore what the pool had given back by then
        // -- evictions past the bound -- and this line is the rest.
        std::fprintf(stderr,
                     "lfric_kokkos_shared_pool: released %zu bytes held at "
                     "finalize\n", held);
      }
    });
    return true;
  }();
  (void)registered;
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

  void *pointer = nullptr;
  bool reused = false;

  if (pooling()) {
    ensure_finalize_hook();
    const auto found = spares.find(nbytes);
    if (found != spares.end() && !found->second.empty()) {
      pointer = found->second.back();
      found->second.pop_back();
      pool_bytes -= nbytes;
      pool_reuses += 1;
      reused = true;
    }
  }

  if (pointer == nullptr) {
    pointer = Kokkos::kokkos_malloc<Kokkos::SharedSpace>(nbytes);
  }

  if (pointer != nullptr) {
    // Advice is a property of the pages, and the pages have not moved: a
    // block re-issued from the free list was advised when it was first
    // allocated and carries that advice still. Advising it again would cost
    // two cudaMemAdvise calls and a prefetch for nothing, and would make
    // advised= on the report line count re-issues rather than blocks. So the
    // counter is the statement: advised= never exceeds the number of blocks
    // this allocator has taken from the driver.
    if (!reused) {
      advise(pointer, nbytes);
    }
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
  live_bytes -= nbytes;
  live_blocks -= 1;

  // With the pool on, and room in it, "freed" means the storage is where the
  // next request of exactly this size will find it. The caller cannot tell:
  // it gets 1 either way, which says "this allocator owned the pointer and
  // the caller must not DEALLOCATE it", and that is as true of a pooled block
  // as of a freed one. is_finalized is checked because a block released after
  // the finalize hook has emptied the pool must go back to the driver rather
  // than into a pool nothing will empty again.
  //
  // One consequence to keep in view: the pool makes an address far more
  // likely to be handed out twice in a run, so a stale pointer to a released
  // field now names live storage belonging to somebody else instead of
  // faulting. That hazard is not new -- the driver reuses addresses too --
  // and what answers it is kokkos_memory_mod's release token, which makes a
  // claim and its release one transaction and refuses a release whose token
  // does not match. The pool raises the odds; the token is the defence.
  if (pooling() && !Kokkos::is_finalized() &&
      pool_bytes + nbytes <= pool_limit()) {
    spares[nbytes].push_back(pointer);
    pool_bytes += nbytes;
    if (pool_bytes > pool_peak) {
      pool_peak = pool_bytes;
    }
    pool_recycled += 1;
    return 1;
  }

  Kokkos::kokkos_free<Kokkos::SharedSpace>(pointer);
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

// Returns 1 if this allocator issued the pointer and has not yet reclaimed it,
// and 0 otherwise. The index query above cannot answer this: it returns zero
// both for a pointer the allocator never issued and for one it issued without
// a registry entry owning it -- the state of every block that came from
// kokkos_shared_allocate rather than kokkos_shared_claim -- and a caller that
// needs to know whether an address is shared-space memory has to tell those
// apart. kokkos_reduce_mod is such a caller: a reduction may only be launched
// over a pointer a device View can be taken of.
extern "C" int lfric_kokkos_shared_owns(void *pointer)
{
  if (pointer == nullptr) {
    return 0;
  }
  return issued.find(pointer) != issued.end() ? 1 : 0;
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
               "%zu bytes live in %zu blocks, advice=%s advised=%zu refused=%zu "
               "pool=%s reuses=%zu recycled=%zu held_peak=%zu released=%zu\n",
               peak_bytes, live_bytes, live_blocks,
               advice_name(), advised_blocks, advice_failures,
               pooling() ? "on" : "off", pool_reuses, pool_recycled,
               pool_peak, pool_released);
}

// The free list's counters, for the Fortran face. Three questions a caller
// can ask without parsing the report line: did the pool fire at all, how many
// requests did it serve, and how much storage is it holding right now. The
// unit-test suite is built without USE_KOKKOS and so links none of this; what
// it asserts through kokkos_memory_mod is that a Fortran-only build answers
// "off, none, nothing", which is the same statement the byte counters make.
extern "C" int lfric_kokkos_shared_pool_enabled()
{
  return pooling() ? 1 : 0;
}

extern "C" std::size_t lfric_kokkos_shared_pool_reuses()
{
  return pool_reuses;
}

extern "C" std::size_t lfric_kokkos_shared_pool_held_bytes()
{
  return pool_bytes;
}

#endif
