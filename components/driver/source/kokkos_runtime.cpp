//----------------------------------------------------------------------------
// (C) Crown copyright 2026 Met Office. All rights reserved.
// The file LICENCE, distributed with this code, contains details of the terms
// under which the code may be used.
//----------------------------------------------------------------------------
//
// The C++ half of driver_kokkos_mod: the only place in an LFRic build that
// starts and stops the Kokkos runtime.
//
// USE_KOKKOS reaches this file through CXXFLAGS rather than through
// PRE_PROCESS_MACROS, because the C++ rule in compile.mk passes CXXFLAGS and
// the Fortran preprocessor macros are a separate list. Without it this is an
// empty translation unit, which is what a Fortran-only build wants: the file is
// still extracted and compiled, but it needs no Kokkos headers and contributes
// no symbols.

#ifdef USE_KOKKOS

#include <Kokkos_Core.hpp>

// Both guards are defensive rather than expected. The driver calls each of
// these exactly once, but a double call would abort inside Kokkos, and an
// abort during model start-up is far harder to read than doing nothing.

extern "C" void lfric_kokkos_initialise()
{
  if (!Kokkos::is_initialized() && !Kokkos::is_finalized()) {
    Kokkos::initialize();
  }
}

extern "C" void lfric_kokkos_finalise()
{
  if (Kokkos::is_initialized()) {
    Kokkos::finalize();
  }
}

#endif
