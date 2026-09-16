!-----------------------------------------------------------------------------
! (C) Crown copyright 2026 Met Office. All rights reserved.
! The file LICENCE, distributed with this code, contains details of the terms
! under which the code may be used.
!-----------------------------------------------------------------------------

!> @brief Runs a whole-field reduction where the field's storage already is.

!> @details The hand-written PSyKAl-lite reductions in the science component
!>          -- the two inner products over an r_solver field and the min/max
!>          over a real64 field -- walk every owned degree of freedom on the
!>          host. When the field's storage came from Kokkos::SharedSpace and
!>          the model is stepping on a card, that walk faults the field back
!>          across the bus: those three sites hold 52% of a C48 run's CPU page
!>          faults (phase 7, task B9). This module offers each of them a
!>          Kokkos reduction over the same memory instead, so that only the
!>          scalar crosses.
!>
!>          Each routine is a *request*, not a command: it returns .true. when
!>          it computed the answer and .false. when the caller must run its own
!>          loop. It refuses when the knob is off, when the build has no
!>          Kokkos, when the runtime is not up, when the data is not storage
!>          the shared allocator issued -- a block that fell back to ALLOCATE
!>          is host memory a device view cannot be taken of -- and when there
!>          is no dof to reduce over, whose identity element the two sides
!>          spell differently. Every caller therefore keeps its Fortran loop,
!>          and a build without USE_KOKKOS runs exactly the arithmetic it ran
!>          before this module existed.
!>
!>          Why here and not in the callers: the PSyKAl-lite routines are
!>          lowercase .f90 and compile.mk passes PRE_PROCESS_MACROS only to
!>          the .F90 rule, so USE_KOKKOS is invisible inside them. Rather than
!>          rename shipped science source, every conditional lives in this
!>          .F90 and the callers branch at run time on a logical -- the
!>          arrangement kokkos_memory_mod and driver_kokkos_mod already use.
!>
!>          Accumulation: the Fortran accumulates an r_solver inner product in
!>          r_double, and so does the C++, element by element. At one thread
!>          Kokkos's Serial and OpenMP backends walk a range in ascending
!>          order, which is the Fortran's order, so a host build at one thread
!>          is expected to agree bit for bit. On a card the reduction is a
!>          tree, and the answer is judged against the measured Fortran
!>          envelope as the atomic regions already are.
!>
!>          The knob is LFRIC_KOKKOS_LITE_REDUCE, read once, off unless it is
!>          set to 1. kokkos_reduce_report writes what it resolved to and how
!>          many reductions were launched and refused.
!>
module kokkos_reduce_mod

  ! real32 and real64 are the kinds the science component's reductions are
  ! instantiated for: r_solver is one of them (R_SOLVER_PRECISION) and the
  ! min/max built-in is always real64. They come from iso_fortran_env rather
  ! than constants_mod because constants_mod imports them without exporting
  ! them.
  use, intrinsic :: iso_fortran_env, only : real32, real64, error_unit

  use, intrinsic :: iso_c_binding,   only : c_ptr, c_size_t, c_double,       &
                                            c_loc

  use constants_mod,                 only : i_def, i_long, l_def, str_def

  use kokkos_memory_mod,             only : kokkos_shared_owns

  implicit none

  private

  public :: kokkos_reduce_enabled,          &
            kokkos_reduce_innerproduct_x,   &
            kokkos_reduce_innerproduct_y,   &
            kokkos_reduce_min_max,          &
            kokkos_reduce_calls,            &
            kokkos_reduce_fallbacks,        &
            kokkos_reduce_report

  !> The environment variable that turns the device reductions on. Off unless
  !> it is exactly '1', because the lever is measured before it is a default.
  character(*), parameter :: reduce_variable = 'LFRIC_KOKKOS_LITE_REDUCE'

  !> Whether the environment has been consulted yet.
  logical(l_def), save :: mode_decided = .false._l_def

  !> Whether the knob asked for the device reductions. Distinct from the
  !> answer below: a build without Kokkos can be asked and cannot serve, and
  !> the report line has to be able to say so rather than look like a knob
  !> nobody set.
  logical(l_def), save :: mode_requested = .false._l_def

  !> Whether the device reductions are available and asked for. Only
  !> meaningful once mode_decided is true.
  logical(l_def), save :: mode_on = .false._l_def

  !> Reductions this module computed.
  integer(i_long), save :: n_calls = 0_i_long

  !> Reductions this module was asked for, with the knob on, and refused --
  !> data the shared allocator had not issued, or nothing to reduce over. A
  !> non-zero count with a null measurement is the first thing to read: it
  !> says the lever was never exercised rather than that it did not pay.
  integer(i_long), save :: n_fallbacks = 0_i_long

#ifdef USE_KOKKOS
  interface

    !> sum over the first ndofs elements of real(x, r_double)**2.
    function lfric_kokkos_reduce_innerproduct_x_real32( x, ndofs )           &
             bind(c, name='lfric_kokkos_reduce_innerproduct_x_real32')       &
             result(total)
      import :: c_ptr, c_size_t, c_double
      implicit none
      type(c_ptr),       value, intent(in) :: x
      integer(c_size_t), value, intent(in) :: ndofs
      real(c_double) :: total
    end function lfric_kokkos_reduce_innerproduct_x_real32

    function lfric_kokkos_reduce_innerproduct_x_real64( x, ndofs )           &
             bind(c, name='lfric_kokkos_reduce_innerproduct_x_real64')       &
             result(total)
      import :: c_ptr, c_size_t, c_double
      implicit none
      type(c_ptr),       value, intent(in) :: x
      integer(c_size_t), value, intent(in) :: ndofs
      real(c_double) :: total
    end function lfric_kokkos_reduce_innerproduct_x_real64

    !> sum over the first ndofs elements of
    !> real(x, r_double) * real(y, r_double).
    function lfric_kokkos_reduce_innerproduct_y_real32( x, y, ndofs )        &
             bind(c, name='lfric_kokkos_reduce_innerproduct_y_real32')       &
             result(total)
      import :: c_ptr, c_size_t, c_double
      implicit none
      type(c_ptr),       value, intent(in) :: x
      type(c_ptr),       value, intent(in) :: y
      integer(c_size_t), value, intent(in) :: ndofs
      real(c_double) :: total
    end function lfric_kokkos_reduce_innerproduct_y_real32

    function lfric_kokkos_reduce_innerproduct_y_real64( x, y, ndofs )        &
             bind(c, name='lfric_kokkos_reduce_innerproduct_y_real64')       &
             result(total)
      import :: c_ptr, c_size_t, c_double
      implicit none
      type(c_ptr),       value, intent(in) :: x
      type(c_ptr),       value, intent(in) :: y
      integer(c_size_t), value, intent(in) :: ndofs
      real(c_double) :: total
    end function lfric_kokkos_reduce_innerproduct_y_real64

    !> The smallest and largest of the first ndofs elements, in one pass.
    subroutine lfric_kokkos_reduce_min_max_real64( x, ndofs,                 &
                                                   smallest, largest )       &
               bind(c, name='lfric_kokkos_reduce_min_max_real64')
      import :: c_ptr, c_size_t, c_double
      implicit none
      type(c_ptr),       value, intent(in)  :: x
      integer(c_size_t), value, intent(in)  :: ndofs
      real(c_double),           intent(out) :: smallest
      real(c_double),           intent(out) :: largest
    end subroutine lfric_kokkos_reduce_min_max_real64

  end interface
#endif

  !> @brief Reduces real(x(df), r_double)**2 over the owned dofs of a field.
  interface kokkos_reduce_innerproduct_x
    module procedure kokkos_reduce_innerproduct_x_real32
    module procedure kokkos_reduce_innerproduct_x_real64
  end interface kokkos_reduce_innerproduct_x

  !> @brief Reduces real(x(df), r_double) * real(y(df), r_double) over the
  !>        owned dofs of two fields.
  interface kokkos_reduce_innerproduct_y
    module procedure kokkos_reduce_innerproduct_y_real32
    module procedure kokkos_reduce_innerproduct_y_real64
  end interface kokkos_reduce_innerproduct_y

contains

  !> @brief Reports whether the device reductions are on for this run.
  !> @details False in a build without USE_KOKKOS, whatever the environment
  !>          says, and false unless LFRIC_KOKKOS_LITE_REDUCE is exactly '1'.
  !>          The environment is read once, as the shared allocator's settings
  !>          are: a knob that changed under a run would make two timesteps of
  !>          one run incomparable.
  !> @return enabled   Whether a reduction request can be served.
  function kokkos_reduce_enabled() result(enabled)

    implicit none

    logical(l_def) :: enabled

    character(str_def) :: setting
    integer            :: status

    if ( .not. mode_decided ) then
      setting = ''
      call get_environment_variable( reduce_variable, setting, status=status )
      ! Not set, or too long to be the one value that means anything here.
      if ( status /= 0 ) setting = ''
      mode_requested = ( trim(adjustl(setting)) == '1' )
#ifdef USE_KOKKOS
      mode_on = mode_requested
#else
      mode_on = .false._l_def
#endif
      mode_decided = .true._l_def
    end if

    enabled = mode_on

  end function kokkos_reduce_enabled

  !> @brief Reduces the square of a 32-bit real field over its owned dofs.
  !> @details Returns .false. without touching 'answer' when the caller must
  !>          run its own loop: the knob off, the storage not the shared
  !>          allocator's, or no dof to reduce over.
  !> @param [in] data     The field's data, as its proxy holds it.
  !> @param [in] ndofs    Owned degrees of freedom, the loop's upper bound.
  !> @param [out] answer  The sum, accumulated in 64-bit real.
  !> @return handled      Whether 'answer' was computed.
  function kokkos_reduce_innerproduct_x_real32( data, ndofs, answer )        &
           result(handled)

    implicit none

    real(real32), pointer, intent(in)  :: data( : )
    integer(i_def),        intent(in)  :: ndofs
    real(real64),          intent(out) :: answer
    logical(l_def) :: handled

    handled = .false._l_def
    answer  = 0.0_real64

    if ( .not. servable( ndofs ) ) return
    if ( .not. associated(data) ) return
    if ( .not. long_enough( size(data, kind=i_def), ndofs ) ) return
    if ( .not. kokkos_shared_owns( data ) ) then
      n_fallbacks = n_fallbacks + 1_i_long
      return
    end if

#ifdef USE_KOKKOS
    answer = real( lfric_kokkos_reduce_innerproduct_x_real32(                &
                       c_loc(data(1)), int(ndofs, c_size_t) ), real64 )
    n_calls = n_calls + 1_i_long
    handled = .true._l_def
#endif

  end function kokkos_reduce_innerproduct_x_real32

  !> @brief Reduces the square of a 64-bit real field over its owned dofs.
  !> @details As kokkos_reduce_innerproduct_x_real32, for the other real kind,
  !>          which is what an r_solver of real64 presents.
  !> @param [in] data     The field's data, as its proxy holds it.
  !> @param [in] ndofs    Owned degrees of freedom, the loop's upper bound.
  !> @param [out] answer  The sum, accumulated in 64-bit real.
  !> @return handled      Whether 'answer' was computed.
  function kokkos_reduce_innerproduct_x_real64( data, ndofs, answer )        &
           result(handled)

    implicit none

    real(real64), pointer, intent(in)  :: data( : )
    integer(i_def),        intent(in)  :: ndofs
    real(real64),          intent(out) :: answer
    logical(l_def) :: handled

    handled = .false._l_def
    answer  = 0.0_real64

    if ( .not. servable( ndofs ) ) return
    if ( .not. associated(data) ) return
    if ( .not. long_enough( size(data, kind=i_def), ndofs ) ) return
    if ( .not. kokkos_shared_owns( data ) ) then
      n_fallbacks = n_fallbacks + 1_i_long
      return
    end if

#ifdef USE_KOKKOS
    answer = real( lfric_kokkos_reduce_innerproduct_x_real64(                &
                       c_loc(data(1)), int(ndofs, c_size_t) ), real64 )
    n_calls = n_calls + 1_i_long
    handled = .true._l_def
#endif

  end function kokkos_reduce_innerproduct_x_real64

  !> @brief Reduces the product of two 32-bit real fields over their owned
  !>        dofs.
  !> @details Both fields must be the shared allocator's, because one launch
  !>          reads both.
  !> @param [in] data1    The first field's data, as its proxy holds it.
  !> @param [in] data2    The second field's data.
  !> @param [in] ndofs    Owned degrees of freedom, the loop's upper bound.
  !> @param [out] answer  The sum, accumulated in 64-bit real.
  !> @return handled      Whether 'answer' was computed.
  function kokkos_reduce_innerproduct_y_real32( data1, data2, ndofs,         &
                                                answer ) result(handled)

    implicit none

    real(real32), pointer, intent(in)  :: data1( : )
    real(real32), pointer, intent(in)  :: data2( : )
    integer(i_def),        intent(in)  :: ndofs
    real(real64),          intent(out) :: answer
    logical(l_def) :: handled

    handled = .false._l_def
    answer  = 0.0_real64

    if ( .not. servable( ndofs ) ) return
    if ( .not. ( associated(data1) .and. associated(data2) ) ) return
    if ( .not. long_enough( size(data1, kind=i_def), ndofs ) ) return
    if ( .not. long_enough( size(data2, kind=i_def), ndofs ) ) return
    if ( .not. ( kokkos_shared_owns( data1 ) .and.                           &
                 kokkos_shared_owns( data2 ) ) ) then
      n_fallbacks = n_fallbacks + 1_i_long
      return
    end if

#ifdef USE_KOKKOS
    answer = real( lfric_kokkos_reduce_innerproduct_y_real32(                &
                       c_loc(data1(1)), c_loc(data2(1)),                     &
                       int(ndofs, c_size_t) ), real64 )
    n_calls = n_calls + 1_i_long
    handled = .true._l_def
#endif

  end function kokkos_reduce_innerproduct_y_real32

  !> @brief Reduces the product of two 64-bit real fields over their owned
  !>        dofs.
  !> @details As kokkos_reduce_innerproduct_y_real32, for the other real kind.
  !> @param [in] data1    The first field's data, as its proxy holds it.
  !> @param [in] data2    The second field's data.
  !> @param [in] ndofs    Owned degrees of freedom, the loop's upper bound.
  !> @param [out] answer  The sum, accumulated in 64-bit real.
  !> @return handled      Whether 'answer' was computed.
  function kokkos_reduce_innerproduct_y_real64( data1, data2, ndofs,         &
                                                answer ) result(handled)

    implicit none

    real(real64), pointer, intent(in)  :: data1( : )
    real(real64), pointer, intent(in)  :: data2( : )
    integer(i_def),        intent(in)  :: ndofs
    real(real64),          intent(out) :: answer
    logical(l_def) :: handled

    handled = .false._l_def
    answer  = 0.0_real64

    if ( .not. servable( ndofs ) ) return
    if ( .not. ( associated(data1) .and. associated(data2) ) ) return
    if ( .not. long_enough( size(data1, kind=i_def), ndofs ) ) return
    if ( .not. long_enough( size(data2, kind=i_def), ndofs ) ) return
    if ( .not. ( kokkos_shared_owns( data1 ) .and.                           &
                 kokkos_shared_owns( data2 ) ) ) then
      n_fallbacks = n_fallbacks + 1_i_long
      return
    end if

#ifdef USE_KOKKOS
    answer = real( lfric_kokkos_reduce_innerproduct_y_real64(                &
                       c_loc(data1(1)), c_loc(data2(1)),                     &
                       int(ndofs, c_size_t) ), real64 )
    n_calls = n_calls + 1_i_long
    handled = .true._l_def
#endif

  end function kokkos_reduce_innerproduct_y_real64

  !> @brief Reduces the smallest and largest value of a 64-bit real field over
  !>        its owned dofs, in one pass.
  !> @details Returns .false. without touching either answer when the caller
  !>          must run its own loop. An empty field is one of those cases: the
  !>          Fortran's identity is huge() and the reducer's is infinity, and
  !>          rather than reconcile them the caller keeps its own answer.
  !> @param [in] data       The field's data, as its proxy holds it.
  !> @param [in] ndofs      Owned degrees of freedom, the loop's upper bound.
  !> @param [out] smallest  The smallest value.
  !> @param [out] largest   The largest value.
  !> @return handled        Whether the two answers were computed.
  function kokkos_reduce_min_max( data, ndofs, smallest, largest )           &
           result(handled)

    implicit none

    real(real64), pointer, intent(in)  :: data( : )
    integer(i_def),        intent(in)  :: ndofs
    real(real64),          intent(out) :: smallest
    real(real64),          intent(out) :: largest
    logical(l_def) :: handled

#ifdef USE_KOKKOS
    real(c_double) :: low, high
#endif

    handled  = .false._l_def
    smallest = 0.0_real64
    largest  = 0.0_real64

    if ( .not. servable( ndofs ) ) return
    if ( .not. associated(data) ) return
    if ( .not. long_enough( size(data, kind=i_def), ndofs ) ) return
    if ( .not. kokkos_shared_owns( data ) ) then
      n_fallbacks = n_fallbacks + 1_i_long
      return
    end if

#ifdef USE_KOKKOS
    call lfric_kokkos_reduce_min_max_real64( c_loc(data(1)),                 &
                                             int(ndofs, c_size_t),           &
                                             low, high )
    smallest = real( low, real64 )
    largest  = real( high, real64 )
    n_calls  = n_calls + 1_i_long
    handled  = .true._l_def
#endif

  end function kokkos_reduce_min_max

  !> @brief Reports how many reductions this module has computed.
  !> @return calls   Reductions launched.
  function kokkos_reduce_calls() result(calls)

    implicit none

    integer(i_long) :: calls

    calls = n_calls

  end function kokkos_reduce_calls

  !> @brief Reports how many reduction requests this module has refused with
  !>        the knob on.
  !> @return fallbacks   Requests the caller had to serve itself.
  function kokkos_reduce_fallbacks() result(fallbacks)

    implicit none

    integer(i_long) :: fallbacks

    fallbacks = n_fallbacks

  end function kokkos_reduce_fallbacks

  !> @brief Writes what the knob resolved to, and the two counts, to standard
  !>        error.
  !> @details Called from driver_kokkos_mod's finalise, beside the shared
  !>          allocator's report and to standard error for the same reason:
  !>          gungho_model finalises the logger before it finalises Kokkos.
  !>
  !>          The line says what the knob resolved to whether or not it was
  !>          set, so that a measurement showing no gain can be checked
  !>          against a knob that never came on before it is read as a lever
  !>          that did not pay. A knob asked for and not served -- a build
  !>          without Kokkos -- says so rather than looking like one nobody
  !>          set.
  subroutine kokkos_reduce_report()

    implicit none

    character(3) :: mode

    if ( kokkos_reduce_enabled() ) then
      mode = 'on '
    else
      mode = 'off'
    end if

    if ( mode_requested .and. .not. mode_on ) then
      write( error_unit, '(A,A,A,I0,A,I0,A)' )                               &
          'lfric_kokkos_reduce: mode=', trim(mode), ' calls=', n_calls,      &
          ' fallbacks=', n_fallbacks,                                        &
          ' (requested on; this build has no Kokkos)'
    else
      write( error_unit, '(A,A,A,I0,A,I0)' )                                 &
          'lfric_kokkos_reduce: mode=', trim(mode), ' calls=', n_calls,      &
          ' fallbacks=', n_fallbacks
    end if
    flush( error_unit )

  end subroutine kokkos_reduce_report

  !> @brief Reports whether a request of this size can be served at all.
  !> @details Two conditions that are the same for every routine here: the
  !>          knob is on, and there is at least one dof to reduce over. An
  !>          empty reduction is refused rather than served because the two
  !>          sides spell its identity differently -- huge() in the Fortran
  !>          loop and infinity in the reducer.
  !>
  !>          A refusal made with the knob on is counted, and one made with it
  !>          off is not: the counters describe a lever that was asked to
  !>          work, so that a null measurement can be told from a knob that
  !>          never came on.
  !> @param [in] ndofs   Owned degrees of freedom the caller asked for.
  !> @return servable    Whether the request can be served.
  function servable( ndofs )

    implicit none

    integer(i_def), intent(in) :: ndofs
    logical(l_def) :: servable

    servable = .false._l_def
    if ( .not. kokkos_reduce_enabled() ) return

    servable = ( ndofs >= 1_i_def )
    if ( .not. servable ) n_fallbacks = n_fallbacks + 1_i_long

  end function servable

  !> @brief Reports whether an array holds the dofs a request named, counting
  !>        a shortfall as a refusal.
  !> @details A field's data array is at least as long as its owned dof count
  !>          -- it carries halo dofs past it -- so a shorter one is a caller
  !>          error rather than a case to serve. It is refused rather than
  !>          read past, and counted, because a reduction over memory beyond
  !>          the array would be wrong on the host and a fault on a card.
  !> @param [in] length  Elements the array holds.
  !> @param [in] ndofs   Elements the request named.
  !> @return long_enough   Whether the request fits.
  function long_enough( length, ndofs )

    implicit none

    integer(i_def), intent(in) :: length
    integer(i_def), intent(in) :: ndofs
    logical(l_def) :: long_enough

    long_enough = ( length >= ndofs )
    if ( .not. long_enough ) n_fallbacks = n_fallbacks + 1_i_long

  end function long_enough

end module kokkos_reduce_mod
