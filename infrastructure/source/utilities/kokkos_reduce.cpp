//----------------------------------------------------------------------------
// (C) Crown copyright 2026 Met Office. All rights reserved.
// The file LICENCE, distributed with this code, contains details of the terms
// under which the code may be used.
//----------------------------------------------------------------------------
//
// The C++ half of kokkos_reduce_mod: whole-field reductions run where the
// field's pages already are.
//
// The three hand-written PSyKAl-lite reductions in lfric_core's science
// component -- the two inner products on r_solver fields and the min/max on a
// real64 field -- traverse every owned degree of freedom of a field on the
// host. When that field's storage came from Kokkos::SharedSpace and the model
// is stepping on a card, the traversal faults the whole field back across the
// bus one page at a time: 30,966, 28,660 and 13,049 CPU page faults in ten
// C48 steps, 52% of the run's, and the largest named part of the host residue
// (phase 7, task B9's fault table). Launching the reduction instead leaves the
// pages where the kernels put them, and only the scalar crosses.
//
// The scalar crosses immediately. A parallel_reduce into a host scalar fences,
// and that is deliberate rather than an oversight: every caller of these
// routines uses the answer in its next statement, through an MPI allreduce.
// FESOM2's port measured deferring a dot product's result as a loss
// (docs/strategy/2026-09-16-fesom-kokkos-lessons.md), so the result is taken
// at once and the design does not repeat that experiment.
//
// Accumulation order, and why the arithmetic is written out rather than left
// to a reducer's default: the Fortran sums 'real(x(df), r_double)**2' into a
// double accumulator, so the C++ converts each element to double and
// accumulates in double too, whatever the element's own precision is. With
// one thread Kokkos's Serial and OpenMP backends walk a RangePolicy in
// ascending order, which is the order the Fortran walks it in, so a host build
// at one thread is expected to agree bit for bit. On a card the reduction is a
// tree and the answer differs in the last places, which is judged against the
// measured Fortran envelope, as the 22 atomic regions already are.
//
// USE_KOKKOS reaches this file through CXXFLAGS rather than through
// PRE_PROCESS_MACROS, for the reason kokkos_runtime.cpp gives: the C++ rule in
// compile.mk passes CXXFLAGS and the Fortran preprocessor macros are a separate
// list. Without it this is an empty translation unit, which is what a
// Fortran-only build wants.

#ifdef USE_KOKKOS

#include <Kokkos_Core.hpp>

#include <cstddef>

namespace {

// The execution space every reduction here runs in, and the memory space its
// views are taken in. On a CUDA build that is CudaSpace, and a SharedSpace
// (managed) pointer is a valid CudaSpace address, which is what lets an
// unmanaged view be taken over storage Fortran allocated. On a host build both
// resolve to the host, and the view is over the same memory the Fortran loop
// would have read.
//
// The memory space is named explicitly rather than left as AnonymousSpace:
// nvcc 13.3 has been seen to mis-infer the address space of an AnonymousSpace
// handle in this toolchain (playbook lesson 90), and there is nothing to gain
// here from a view that could have come from either side.
using ExecSpace = Kokkos::DefaultExecutionSpace;
using MemSpace = ExecSpace::memory_space;

template <typename T>
using ConstView = Kokkos::View<const T *, MemSpace,
                               Kokkos::MemoryTraits<Kokkos::Unmanaged>>;

// sum over df of real(x(df), r_double)**2, in double whatever T is.
template <typename T>
double innerproduct_x(const T *x, std::size_t n)
{
  ConstView<T> view(x, n);
  double total = 0.0;
  Kokkos::parallel_reduce(
      "lfric_kokkos_reduce_innerproduct_x",
      Kokkos::RangePolicy<ExecSpace>(0, n),
      KOKKOS_LAMBDA(const std::size_t df, double &sum) {
        const double value = static_cast<double>(view(df));
        sum += value * value;
      },
      total);
  return total;
}

// sum over df of real(x(df), r_double) * real(y(df), r_double).
template <typename T>
double innerproduct_y(const T *x, const T *y, std::size_t n)
{
  ConstView<T> view_x(x, n);
  ConstView<T> view_y(y, n);
  double total = 0.0;
  Kokkos::parallel_reduce(
      "lfric_kokkos_reduce_innerproduct_y",
      Kokkos::RangePolicy<ExecSpace>(0, n),
      KOKKOS_LAMBDA(const std::size_t df, double &sum) {
        sum += static_cast<double>(view_x(df)) * static_cast<double>(view_y(df));
      },
      total);
  return total;
}

}  // namespace

// Each entry point takes the field's data pointer and the count of owned
// degrees of freedom, and is called only for a pointer the shared allocator
// issued; kokkos_reduce_mod makes both checks before it calls. A zero count is
// refused there too, so that the identity of an empty reduction -- infinity
// for Kokkos::Min, huge() for the Fortran -- never has to be reconciled.
//
// The r_solver kind is a build-time choice (R_SOLVER_PRECISION), so both real
// kinds are instantiated and the Fortran picks; a build with r_solver at
// real128 falls back to its Fortran loop, which no accelerator reduction can
// serve.

extern "C" double lfric_kokkos_reduce_innerproduct_x_real32(const float *x,
                                                            std::size_t ndofs)
{
  return innerproduct_x(x, ndofs);
}

extern "C" double lfric_kokkos_reduce_innerproduct_x_real64(const double *x,
                                                            std::size_t ndofs)
{
  return innerproduct_x(x, ndofs);
}

extern "C" double lfric_kokkos_reduce_innerproduct_y_real32(const float *x,
                                                            const float *y,
                                                            std::size_t ndofs)
{
  return innerproduct_y(x, y, ndofs);
}

extern "C" double lfric_kokkos_reduce_innerproduct_y_real64(const double *x,
                                                            const double *y,
                                                            std::size_t ndofs)
{
  return innerproduct_y(x, y, ndofs);
}

// One pass, two reducers: Kokkos takes several reduction arguments on one
// parallel_reduce, so the field is read once for both answers, as the Fortran
// loop reads it once for both of its thread-local accumulators.
extern "C" void lfric_kokkos_reduce_min_max_real64(const double *x,
                                                   std::size_t ndofs,
                                                   double *smallest,
                                                   double *largest)
{
  ConstView<double> view(x, ndofs);
  double low = 0.0;
  double high = 0.0;
  Kokkos::parallel_reduce(
      "lfric_kokkos_reduce_min_max",
      Kokkos::RangePolicy<ExecSpace>(0, ndofs),
      KOKKOS_LAMBDA(const std::size_t df, double &smaller, double &larger) {
        const double value = view(df);
        if (value < smaller) {
          smaller = value;
        }
        if (value > larger) {
          larger = value;
        }
      },
      Kokkos::Min<double>(low), Kokkos::Max<double>(high));
  *smallest = low;
  *largest = high;
}

#endif
