!-----------------------------------------------------------------------------
! (C) Crown copyright 2026 Met Office. All rights reserved.
! The file LICENCE, distributed with this code, contains details of the terms
! under which the code may be used.
!-----------------------------------------------------------------------------

!> @brief Controls the initialisation and finalisation of the Kokkos runtime

!> @details Generated Kokkos regions require Kokkos to have been initialised
!>          before the first region runs, and finalised afterwards. Kokkos does
!>          not enforce this. An uninitialised region writes
!>          "OpenMP is not initialized" to standard error and then runs anyway,
!>          producing correct results single-threaded, so a missing
!>          initialisation fails no build and no smoke test; it shows up only
!>          as silently lost threading.
!>
!>          The calls belong in the driver rather than in the generated region
!>          because Kokkos must be finalised before MPI, and the main program
!>          calls final_comm -- hence MPI_Finalize -- as its last statement. A
!>          C++ atexit handler would run after that, which Kokkos does not
!>          support. Initialising once here also avoids paying for it per
!>          region.
!>
!>          Compiled without USE_KOKKOS both routines are empty and nothing
!>          links against a C++ runtime, so a Fortran-only build is unchanged.
!>          This follows the USE_XIOS pattern in driver_comm_mod.
!>
module driver_kokkos_mod

  use kokkos_memory_mod, only: kokkos_shared_release_all, &
                               kokkos_shared_report

  implicit none

  private
  public :: init_kokkos, final_kokkos

#ifdef USE_KOKKOS
  interface

    subroutine lfric_kokkos_initialise() bind(c, name='lfric_kokkos_initialise')
    end subroutine lfric_kokkos_initialise

    subroutine lfric_kokkos_finalise() bind(c, name='lfric_kokkos_finalise')
    end subroutine lfric_kokkos_finalise

  end interface
#endif

contains

  !> @brief Initialises the Kokkos runtime.
  !> @details Called after the model communicator exists, so that Kokkos is
  !>          nested inside the lifetime of MPI. Does nothing in a build
  !>          without Kokkos.
  subroutine init_kokkos()

#ifdef USE_KOKKOS
    call lfric_kokkos_initialise()
#endif

  end subroutine init_kokkos

  !> @brief Finalises the Kokkos runtime.
  !> @details Called before the model communicator is destroyed, for the same
  !>          reason. The Kokkos call itself does nothing in a build without
  !>          Kokkos; the release below runs in both.
  !>
  !>          The blocks kokkos_memory_mod is holding are released first,
  !>          because one from shared space cannot be returned after the
  !>          runtime it came from has gone. Anything still pointing at a
  !>          claimed block is dangling afterwards, which is why this happens
  !>          on the way out and nowhere else. The release runs in a build
  !>          without Kokkos too, where the blocks came from ALLOCATE: leaving
  !>          the two builds to reclaim at different points would be a
  !>          difference between them that nothing else in the run needs.
  !>
  !>          The shared-space high-water mark is then reported, to standard
  !>          error because gungho_model.f90 calls final_logger before
  !>          final_kokkos and so has no logger left. Reporting after the
  !>          release rather than before it is what makes the live figures a
  !>          leak check: everything this run claimed has been given back by
  !>          then, so a non-zero live count is a block claimed by something
  !>          the registry did not see. The peak figures survive the release
  !>          and are what say the storage was ever taken from shared space.
  !>          Both must come before Kokkos::finalize, since the counters they
  !>          read live in a translation unit whose state means nothing after
  !>          it.
  subroutine final_kokkos()

    call kokkos_shared_release_all()
    call kokkos_shared_report()

#ifdef USE_KOKKOS
    call lfric_kokkos_finalise()
#endif

  end subroutine final_kokkos

end module driver_kokkos_mod
