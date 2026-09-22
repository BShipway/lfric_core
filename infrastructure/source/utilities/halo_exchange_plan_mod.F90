!-----------------------------------------------------------------------------
! (C) Crown copyright 2026 Met Office. All rights reserved.
! The file LICENCE, distributed with this code, contains details of the terms
! under which the code may be used.
!-----------------------------------------------------------------------------

!> @brief Reads YAXT's own send and receive lists out of an exchange map and
!>        replays a halo exchange from them.

!> @details A halo exchange today is one call to `xt_redist_s_exchange`, and
!>          the profile of a six-rank C144 step says that about half of its
!>          cost is one thing: the host walking the whole owned block and
!>          gathering, for each neighbour, the dofs that neighbour needs. YAXT
!>          does not pack that itself -- it hands the scatter to MPI as an
!>          indexed derived datatype -- so the gather appears inside OpenMPI's
!>          datatype convertor and reads device-resident pages from the CPU.
!>
!>          To gather on the device instead, something has to know which dofs
!>          go to which neighbour. That list exists: YAXT computed it when the
!>          exchange map was built, LFRic keeps the map
!>          (`exchange_map_type%get_xmap`), and YAXT's public interface hands
!>          it back -- `xt_xmap_get_out_iterator` and `xt_xmap_get_in_iterator`
!>          walk the destinations and the sources, and for each one
!>          `xt_xmap_iterator_get_transfer_pos` gives the positions to gather
!>          from or scatter into. Those calls are exported by the `yaxt`
!>          Fortran module itself, so this module needs no C of its own.
!>
!>          That matters for more than convenience. Deriving the lists a second
!>          time from `global_dof_id` and the partition would be new parallel
!>          index arithmetic that could disagree with YAXT in a corner; reading
!>          YAXT's own lists cannot disagree with YAXT. Bit-identity is a
!>          property of where the lists come from, not an argument about how
!>          they were computed.
!>
!>          The positions are positions in the two index lists YAXT was given,
!>          counted from zero. `generate_redistribution_map` gives the source
!>          list offsets `i - 1` and the target list offsets
!>          `i + size(src) - 1` into one flat array holding the owned block
!>          followed by the halo slab, so a send position `p` is element
!>          `p + 1` of `data_with_halos` and a receive position `q` is element
!>          `q + n_owned + 1`. Those two lines are the whole of the mapping,
!>          and `n_owned` is the only thing this module needs from LFRic that
!>          the map does not carry.
!>
!>          This module is the host replay: gather with the lists, exchange
!>          with plain non-blocking MPI, scatter with the lists. It exists
!>          before any device code so that the extraction can be falsified on
!>          its own. `LFRIC_KOKKOS_HALO_SELFCHECK=1` runs YAXT's exchange on a
!>          copy beside it and compares the halo slab word for word, which is
!>          the falsifier the plan asks for and is also the check a first run
!>          on new hardware should make.
!>
!>          Both 32-bit and 64-bit real fields are served, and that is not a
!>          courtesy. The baseline profile of a six-rank C144 step charges
!>          about 11.5% of the step to `field_real32_mod`'s halo exchange and
!>          about 9.6% to `field_real64_mod`'s -- the 32-bit exchange is the
!>          LARGER of the two, on every rank. A 64-bit-only replay would have
!>          left more than half the thing it was built to remove in place, and
!>          an A/B row would have read as a half-sized lever for a reason that
!>          had nothing to do with the approach. The position lists do not
!>          depend on the kind, so one plan serves both; only the buffers and
!>          the MPI datatype differ, and each kind's buffers are allocated the
!>          first time that kind is exchanged.
!>
!>          Integer fields are not served -- they are below the noise in the
!>          same profile -- and a routing of any other type falls back to YAXT.
!>          The count of fallbacks is in the report line, so a measurement
!>          showing no gain can be checked against how often the path was taken
!>          at all.
!
module halo_exchange_plan_mod

  use, intrinsic :: iso_fortran_env, only : real32, real64,                  &
                                            int32, int64, error_unit
  use, intrinsic :: iso_c_binding,   only : c_int

  use constants_mod, only : i_def, l_def, str_def
  use log_mod,       only : log_event, LOG_LEVEL_ERROR

#ifdef NO_MPI
  ! Without MPI there are no neighbours, so there is nothing to plan.
#else
  use lfric_mpi_mod, only : global_mpi, lfric_comm_type
#ifdef LEGACY_MPI
  use mpi,     only : mpi_double_precision, mpi_real, mpi_statuses_ignore, &
                      mpi_isend, mpi_irecv, mpi_waitall
#else
  use mpi_f08, only : mpi_comm, mpi_request, mpi_double_precision, mpi_real, &
                      mpi_statuses_ignore, &
                      mpi_isend, mpi_irecv, mpi_waitall
#endif
  use yaxt,    only : xt_xmap, xt_xmap_iter, xt_redist,           &
                      xt_redist_s_exchange,                       &
                      xt_xmap_get_num_destinations,               &
                      xt_xmap_get_num_sources,                    &
                      xt_xmap_get_out_iterator,                   &
                      xt_xmap_get_in_iterator,                    &
                      xt_xmap_iterator_next,                      &
                      xt_xmap_iterator_get_rank,                  &
                      xt_xmap_iterator_get_transfer_pos,          &
                      xt_xmap_iterator_get_num_transfer_pos,      &
                      xt_xmap_iterator_delete
#endif

  implicit none

  private

  public :: halo_plan_type
  public :: halo_plan_build
  public :: halo_plan_exchange_real32, halo_plan_exchange_real64
  public :: halo_plan_mode_enabled, halo_plan_selfcheck_enabled
  public :: halo_plan_report, halo_plan_count_fallback

  !> The name of the knob that turns the replay on.
  character(*), parameter :: mode_variable = 'LFRIC_KOKKOS_HALO'
  !> The name of the knob that runs YAXT beside it and compares.
  character(*), parameter :: check_variable = 'LFRIC_KOKKOS_HALO_SELFCHECK'
  !> The name of the memory-placement knob this path refuses to share with.
  character(*), parameter :: advice_variable = 'LFRIC_KOKKOS_MEM_ADVICE'
  !> One tag for every message; the exchange is blocking, so only one is in
  !> flight between a pair at a time.
  integer, parameter :: halo_tag = 7241

  !> @details What one exchange needs to know, for one routing at one depth.
  !>          `send_index` and `recv_index` are one-based indices into
  !>          `data_with_halos`, laid out neighbour by neighbour so that each
  !>          neighbour's message is one contiguous slice of the buffer.
  type, public :: halo_plan_type
    private
    !> Whether the lists have been read out of the exchange map yet.
    logical(l_def) :: built = .false._l_def
    !> Whether this plan can serve an exchange at all.
    logical(l_def) :: served = .false._l_def
    !> Dofs owned by this rank; the offset between a receive position and its
    !> index in the flat array.
    integer(i_def) :: n_owned = 0
    !> Number of neighbours sent to, and received from.
    integer(i_def) :: n_dest = 0
    integer(i_def) :: n_src  = 0
    !> Per destination: the neighbour's rank, and its slice of `send_index`.
    integer(i_def), allocatable :: dest_rank(:)
    integer(i_def), allocatable :: dest_start(:)
    integer(i_def), allocatable :: dest_count(:)
    !> Per source: the neighbour's rank, and its slice of `recv_index`.
    integer(i_def), allocatable :: src_rank(:)
    integer(i_def), allocatable :: src_start(:)
    integer(i_def), allocatable :: src_count(:)
    !> The gather list and the scatter list, one-based.
    integer(i_def), allocatable :: send_index(:)
    integer(i_def), allocatable :: recv_index(:)
    !> Whether every receive slice is the contiguous run its own dofs already
    !> occupy, which is what the design's fact 2 predicts and what a device
    !> receive can copy rather than scatter.
    logical(l_def) :: recv_contiguous = .false._l_def
    !> The buffers and the request handles, held by the plan because the plan
    !> is cached and an exchange happens 361 times a step: none of this is
    !> work to allocate again each time. A routing is keyed by Fortran type and
    !> kind (`halo_routing_constructor` takes both), so in practice only one
    !> kind's pair is ever allocated on a given plan; they are allocated on
    !> first use rather than at build so that nothing is paid for a kind this
    !> routing never carries.
    real(real64), allocatable :: send_buffer(:)
    real(real64), allocatable :: recv_buffer(:)
    real(real32), allocatable :: send_buffer32(:)
    real(real32), allocatable :: recv_buffer32(:)
#if defined(NO_MPI) || defined(LEGACY_MPI)
    integer, allocatable :: requests(:)
#else
    type(mpi_request), allocatable :: requests(:)
#endif
  contains
    procedure, public :: is_built
    procedure, public :: is_served
    procedure, public :: send_dofs
    procedure, public :: recv_dofs
    procedure, public :: neighbours
    procedure, public :: receives_are_contiguous
  end type halo_plan_type

  !> The knobs, read once each.
  logical(l_def), save :: mode_decided = .false._l_def
  logical(l_def), save :: mode_on      = .false._l_def
  logical(l_def), save :: check_decided = .false._l_def
  logical(l_def), save :: check_on      = .false._l_def

  !> What the path did, for the report line.
  integer(i_def), save :: n_plans     = 0
  integer(i_def), save :: n_exchanges = 0
  integer(i_def), save :: n_fallbacks = 0
  integer(i_def), save :: n_checked   = 0
  integer(i_def), save :: n_mismatch  = 0

contains

  !> @brief Reports whether this plan has read its lists yet.
  !> @param[in] self  The plan.
  !> @return built    Whether the lists have been read.
  function is_built(self) result(built)

    implicit none

    class(halo_plan_type), intent(in) :: self
    logical(l_def) :: built

    built = self%built

  end function is_built

  !> @brief Reports whether this plan can serve an exchange.
  !> @param[in] self  The plan.
  !> @return served   Whether the replay may be used for this routing.
  function is_served(self) result(served)

    implicit none

    class(halo_plan_type), intent(in) :: self
    logical(l_def) :: served

    served = self%served

  end function is_served

  !> @brief The number of dofs this rank sends in one exchange.
  !> @param[in] self  The plan.
  !> @return dofs     Total dofs gathered, over every neighbour.
  function send_dofs(self) result(dofs)

    implicit none

    class(halo_plan_type), intent(in) :: self
    integer(i_def) :: dofs

    dofs = 0
    if ( allocated(self%send_index) ) dofs = size(self%send_index)

  end function send_dofs

  !> @brief The number of dofs this rank receives in one exchange.
  !> @param[in] self  The plan.
  !> @return dofs     Total dofs scattered, over every neighbour.
  function recv_dofs(self) result(dofs)

    implicit none

    class(halo_plan_type), intent(in) :: self
    integer(i_def) :: dofs

    dofs = 0
    if ( allocated(self%recv_index) ) dofs = size(self%recv_index)

  end function recv_dofs

  !> @brief The number of neighbours this rank talks to.
  !> @param[in] self  The plan.
  !> @return n        The larger of the destination and source counts.
  function neighbours(self) result(n)

    implicit none

    class(halo_plan_type), intent(in) :: self
    integer(i_def) :: n

    n = max(self%n_dest, self%n_src)

  end function neighbours

  !> @brief Whether every receive slice is already contiguous.
  !> @param[in] self      The plan.
  !> @return contiguous   Whether a receive can be a copy rather than a
  !>                      scatter.
  function receives_are_contiguous(self) result(contiguous)

    implicit none

    class(halo_plan_type), intent(in) :: self
    logical(l_def) :: contiguous

    contiguous = self%recv_contiguous

  end function receives_are_contiguous

#ifdef NO_MPI

  !> @brief Builds nothing: a serial build has no neighbours.
  !> @param[in,out] self     The plan.
  !> @param[in]     xmap     Ignored; an integer in a serial build.
  !> @param[in]     n_owned  Ignored.
  subroutine halo_plan_build( self, xmap, n_owned )

    implicit none

    type(halo_plan_type), intent(inout) :: self
    integer(i_def),       intent(in)    :: xmap
    integer(i_def),       intent(in)    :: n_owned

    self%built   = .true._l_def
    self%served  = .false._l_def
    self%n_owned = n_owned
    if ( xmap /= 0 ) self%n_owned = n_owned

  end subroutine halo_plan_build

  !> @brief Exchanges nothing: a serial build has no neighbours.
  !> @param[in,out] data_with_halos  Ignored.
  !> @param[in,out] self             Ignored.
  !> @param[in]     redist           Ignored; an integer in a serial build.
  subroutine halo_plan_exchange_real64( data_with_halos, self, redist )

    implicit none

    real(real64),         intent(inout), pointer :: data_with_halos(:)
    type(halo_plan_type), intent(inout)          :: self
    integer(i_def),       intent(in),    pointer :: redist

    if ( associated(data_with_halos) .and. self%built .and.                  &
         associated(redist) ) continue

  end subroutine halo_plan_exchange_real64

  !> @brief Exchanges nothing: a serial build has no neighbours.
  !> @param[in,out] data_with_halos  Ignored.
  !> @param[in,out] self             Ignored.
  !> @param[in]     redist           Ignored; an integer in a serial build.
  subroutine halo_plan_exchange_real32( data_with_halos, self, redist )

    implicit none

    real(real32),         intent(inout), pointer :: data_with_halos(:)
    type(halo_plan_type), intent(inout)          :: self
    integer(i_def),       intent(in),    pointer :: redist

    if ( associated(data_with_halos) .and. self%built .and.                  &
         associated(redist) ) continue

  end subroutine halo_plan_exchange_real32

#else

  !> @brief Reads the send and receive lists out of an exchange map.
  !> @details Two passes over each iterator: the first counts, so that the
  !>          arrays can be allocated once at the size they will need; the
  !>          second fills them. An iterator is cheap and a plan is built once
  !>          per routing and depth, so two passes are paid for by never
  !>          growing an array.
  !>
  !>          A rank with no destinations and no sources -- a one-rank run --
  !>          builds a plan that is marked built and not served, and the
  !>          caller keeps using YAXT. That is deliberate: a plan that served
  !>          an empty exchange would make a one-rank A/B row measure this
  !>          module's own branch rather than the exchange.
  !> @param[in,out] self     The plan to fill.
  !> @param[in]     xmap     The exchange map the redistribution was built
  !>                         from.
  !> @param[in]     n_owned  The number of owned dofs, which is where the halo
  !>                         slab starts in the flat array.
  subroutine halo_plan_build( self, xmap, n_owned )

    implicit none

    type(halo_plan_type), intent(inout) :: self
    type(xt_xmap),        intent(in)    :: xmap
    integer(i_def),       intent(in)    :: n_owned

    type(xt_xmap_iter)         :: iter
    integer(c_int), pointer    :: pos(:)
    integer(i_def)             :: i, k, n, cursor
    integer(i_def)             :: expected

    self%built   = .true._l_def
    self%served  = .false._l_def
    self%n_owned = n_owned

    self%n_dest = int(xt_xmap_get_num_destinations(xmap), i_def)
    self%n_src  = int(xt_xmap_get_num_sources(xmap), i_def)

    if ( self%n_dest <= 0 .and. self%n_src <= 0 ) return

    allocate( self%dest_rank(max(self%n_dest, 1)) )
    allocate( self%dest_start(max(self%n_dest, 1)) )
    allocate( self%dest_count(max(self%n_dest, 1)) )
    allocate( self%src_rank(max(self%n_src, 1)) )
    allocate( self%src_start(max(self%n_src, 1)) )
    allocate( self%src_count(max(self%n_src, 1)) )
    self%dest_rank(:)  = -1
    self%dest_start(:) = 1
    self%dest_count(:) = 0
    self%src_rank(:)   = -1
    self%src_start(:)  = 1
    self%src_count(:)  = 0

    ! ---- the send side: positions in the source list, offset i - 1.
    n = 0
    if ( self%n_dest > 0 ) then
      iter = xt_xmap_get_out_iterator(xmap)
      i = 0
      do
        i = i + 1
        if ( i > self%n_dest ) exit
        self%dest_rank(i)  = int(xt_xmap_iterator_get_rank(iter), i_def)
        self%dest_count(i) = int(xt_xmap_iterator_get_num_transfer_pos(iter), &
                                 i_def)
        self%dest_start(i) = n + 1
        n = n + self%dest_count(i)
        if ( .not. xt_xmap_iterator_next(iter) ) exit
      end do
      call xt_xmap_iterator_delete(iter)
    end if
    allocate( self%send_index(max(n, 1)) )
    self%send_index(:) = 1
    if ( self%n_dest > 0 ) then
      iter = xt_xmap_get_out_iterator(xmap)
      i = 0
      do
        i = i + 1
        if ( i > self%n_dest ) exit
        pos => xt_xmap_iterator_get_transfer_pos(iter)
        cursor = self%dest_start(i)
        do k = 1, self%dest_count(i)
          self%send_index(cursor + k - 1) = int(pos(k), i_def) + 1
        end do
        if ( .not. xt_xmap_iterator_next(iter) ) exit
      end do
      call xt_xmap_iterator_delete(iter)
    end if

    ! ---- the receive side: positions in the target list, offset
    !      i + n_owned - 1.
    n = 0
    if ( self%n_src > 0 ) then
      iter = xt_xmap_get_in_iterator(xmap)
      i = 0
      do
        i = i + 1
        if ( i > self%n_src ) exit
        self%src_rank(i)  = int(xt_xmap_iterator_get_rank(iter), i_def)
        self%src_count(i) = int(xt_xmap_iterator_get_num_transfer_pos(iter), &
                                i_def)
        self%src_start(i) = n + 1
        n = n + self%src_count(i)
        if ( .not. xt_xmap_iterator_next(iter) ) exit
      end do
      call xt_xmap_iterator_delete(iter)
    end if
    allocate( self%recv_index(max(n, 1)) )
    self%recv_index(:) = 1
    self%recv_contiguous = .true._l_def
    if ( self%n_src > 0 ) then
      iter = xt_xmap_get_in_iterator(xmap)
      i = 0
      do
        i = i + 1
        if ( i > self%n_src ) exit
        pos => xt_xmap_iterator_get_transfer_pos(iter)
        cursor = self%src_start(i)
        do k = 1, self%src_count(i)
          self%recv_index(cursor + k - 1) = int(pos(k), i_def) + n_owned + 1
        end do
        ! Contiguity is a property worth knowing, not one worth relying on:
        ! it decides whether a device receive can be a copy, and it is
        ! measured here rather than assumed from the design.
        do k = 2, self%src_count(i)
          expected = self%recv_index(cursor + k - 2) + 1
          if ( self%recv_index(cursor + k - 1) /= expected ) then
            self%recv_contiguous = .false._l_def
            exit
          end if
        end do
        if ( .not. xt_xmap_iterator_next(iter) ) exit
      end do
      call xt_xmap_iterator_delete(iter)
    end if
    allocate( self%requests(max(self%n_dest + self%n_src, 1)) )

    self%served = .true._l_def
    n_plans = n_plans + 1

  end subroutine halo_plan_build

  !> @brief Replays one halo exchange from the plan's lists.
  !> @details Gather, post every receive, post every send, wait, scatter.
  !>          The receives go up first so that a message arriving before its
  !>          receive is posted is the exception rather than the rule, which
  !>          is what keeps the unexpected-message queue empty on a shared
  !>          memory transport.
  !>
  !>          With the selfcheck on, YAXT does the same exchange first on a
  !>          copy of the whole field and the two halo slabs are compared word
  !>          for word. The comparison is bitwise, on a copy, and the field the
  !>          model goes on to use is the replay's -- so a selfcheck run that
  !>          completes with `mismatches=0` says the replay's halo is YAXT's
  !>          halo at every one of the exchanges it took, not at a sample.
  !> @param[in,out] data_with_halos  The field, owned block then halo slab.
  !> @param[in,out] self             The plan; its buffers are written.
  !> @param[in]     redist           YAXT's redistribution for this routing
  !>                                 and depth, used only by the selfcheck.
  subroutine halo_plan_exchange_real64( data_with_halos, self, redist )

    implicit none

    real(real64),         intent(inout), pointer :: data_with_halos(:)
    type(halo_plan_type), intent(inout)          :: self
    type(xt_redist),      intent(in),    pointer :: redist

#ifdef LEGACY_MPI
    integer        :: comm
#else
    type(mpi_comm) :: comm
#endif
    type(lfric_comm_type)      :: lfric_comm
    integer                    :: ierr
    integer(i_def)             :: i, k, first, last, n_words, n_requests
    integer(i_def)             :: mismatches
    real(real64), allocatable, target :: reference(:)

    if ( .not. self%served ) return

    if ( .not. allocated(self%send_buffer) ) then
      allocate( self%send_buffer(max(size(self%send_index), 1)) )
      allocate( self%recv_buffer(max(size(self%recv_index), 1)) )
    end if

    if ( halo_plan_selfcheck_enabled() ) then
      allocate( reference(size(data_with_halos)) )
      reference(:) = data_with_halos(:)
      call xt_redist_s_exchange( redist, reference, reference )
    end if

    ! The communicator is fetched into a local of the right type once, rather
    ! than selected inside each call: a preprocessor branch in the middle of a
    ! continued statement is the kind of thing that compiles here and not
    ! elsewhere.
    lfric_comm = global_mpi%get_comm()
#ifdef LEGACY_MPI
    comm = lfric_comm%get_comm_mpi_val()
#else
    comm%mpi_val = lfric_comm%get_comm_mpi_val()
#endif

    do k = 1, size(self%send_index)
      self%send_buffer(k) = data_with_halos(self%send_index(k))
    end do

    n_requests = 0

    ! The buffer is handed over as a contiguous array SECTION, not as the
    ! first element with a count beside it. The older idiom relies on sequence
    ! association, which mpi_f08's assumed-rank dummies do not provide; a
    ! section says the same thing in a form both bindings accept, and these
    ! sections are contiguous by construction so nothing is copied.
    do i = 1, self%n_src
      n_words = self%src_count(i)
      if ( n_words <= 0 ) cycle
      first = self%src_start(i)
      last  = first + n_words - 1
      n_requests = n_requests + 1
      call mpi_irecv( self%recv_buffer(first:last), int(n_words),             &
                      mpi_double_precision,                                   &
                      int(self%src_rank(i)), halo_tag, comm,                  &
                      self%requests(n_requests), ierr )
    end do

    do i = 1, self%n_dest
      n_words = self%dest_count(i)
      if ( n_words <= 0 ) cycle
      first = self%dest_start(i)
      last  = first + n_words - 1
      n_requests = n_requests + 1
      call mpi_isend( self%send_buffer(first:last), int(n_words),             &
                      mpi_double_precision,                                   &
                      int(self%dest_rank(i)), halo_tag, comm,                 &
                      self%requests(n_requests), ierr )
    end do

    if ( n_requests > 0 ) then
      call mpi_waitall( int(n_requests), self%requests(1:n_requests),        &
                        mpi_statuses_ignore, ierr )
    end if

    do k = 1, size(self%recv_index)
      data_with_halos(self%recv_index(k)) = self%recv_buffer(k)
    end do

    n_exchanges = n_exchanges + 1

    ! The comparison is on BIT PATTERNS, not on values. Two reasons, and both
    ! matter. The claim being tested is bit-identity, and `/=` on reals is not
    ! that test: it is true for two NaNs with the same bits and false for
    ! +0.0 against -0.0. And the build this runs on compiles with
    ! `-finit-real=snan`, so every halo word no exchange has yet written holds
    ! a NaN -- which a value comparison reports as a difference between an
    ! array and an exact copy of itself. Transferring to an integer of the
    ! same width asks the question that was meant.
    if ( allocated(reference) ) then
      mismatches = 0
      do k = self%n_owned + 1, size(data_with_halos)
        if ( transfer(data_with_halos(k), 0_int64) /=                        &
             transfer(reference(k), 0_int64) ) mismatches = mismatches + 1
      end do
      call halo_plan_count_check( mismatches )
      deallocate( reference )
    end if

  end subroutine halo_plan_exchange_real64

  !> @brief Replays one 32-bit halo exchange from the plan's lists.
  !> @details The twin of `halo_plan_exchange_real64`, and deliberately a twin
  !>          rather than a generic: `halo_comms_mod` itself writes
  !>          `perform_halo_exchange_real32`, `_real64` and `_int32` out in
  !>          full, and a kind-generic replay here would be the only thing in
  !>          this part of the infrastructure that was not.
  !>
  !>          The plan is shared with the 64-bit routine because the position
  !>          lists do not depend on the kind. Only the buffers and the MPI
  !>          datatype differ. In practice a given plan sees one kind, since a
  !>          `halo_routing` is keyed by Fortran type and kind; the allocation
  !>          test below is what makes that a fact about the run rather than an
  !>          assumption in the code.
  !>
  !>          This kind is the larger half of the thing being removed: the
  !>          baseline charges about 11.5% of a six-rank C144 step to the
  !>          32-bit exchange against about 9.6% to the 64-bit one.
  !> @param[in,out] data_with_halos  The field, owned block then halo slab.
  !> @param[in,out] self             The plan; its 32-bit buffers are written.
  !> @param[in]     redist           YAXT's redistribution for this routing
  !>                                 and depth, used only by the selfcheck.
  subroutine halo_plan_exchange_real32( data_with_halos, self, redist )

    implicit none

    real(real32),         intent(inout), pointer :: data_with_halos(:)
    type(halo_plan_type), intent(inout)          :: self
    type(xt_redist),      intent(in),    pointer :: redist

#ifdef LEGACY_MPI
    integer        :: comm
#else
    type(mpi_comm) :: comm
#endif
    type(lfric_comm_type)      :: lfric_comm
    integer                    :: ierr
    integer(i_def)             :: i, k, first, last, n_words, n_requests
    integer(i_def)             :: mismatches
    real(real32), allocatable, target :: reference(:)

    if ( .not. self%served ) return

    if ( .not. allocated(self%send_buffer32) ) then
      allocate( self%send_buffer32(max(size(self%send_index), 1)) )
      allocate( self%recv_buffer32(max(size(self%recv_index), 1)) )
    end if

    if ( halo_plan_selfcheck_enabled() ) then
      allocate( reference(size(data_with_halos)) )
      reference(:) = data_with_halos(:)
      call xt_redist_s_exchange( redist, reference, reference )
    end if

    lfric_comm = global_mpi%get_comm()
#ifdef LEGACY_MPI
    comm = lfric_comm%get_comm_mpi_val()
#else
    comm%mpi_val = lfric_comm%get_comm_mpi_val()
#endif

    do k = 1, size(self%send_index)
      self%send_buffer32(k) = data_with_halos(self%send_index(k))
    end do

    n_requests = 0

    do i = 1, self%n_src
      n_words = self%src_count(i)
      if ( n_words <= 0 ) cycle
      first = self%src_start(i)
      last  = first + n_words - 1
      n_requests = n_requests + 1
      call mpi_irecv( self%recv_buffer32(first:last), int(n_words),            &
                      mpi_real,                                                &
                      int(self%src_rank(i)), halo_tag, comm,                   &
                      self%requests(n_requests), ierr )
    end do

    do i = 1, self%n_dest
      n_words = self%dest_count(i)
      if ( n_words <= 0 ) cycle
      first = self%dest_start(i)
      last  = first + n_words - 1
      n_requests = n_requests + 1
      call mpi_isend( self%send_buffer32(first:last), int(n_words),            &
                      mpi_real,                                                &
                      int(self%dest_rank(i)), halo_tag, comm,                  &
                      self%requests(n_requests), ierr )
    end do

    if ( n_requests > 0 ) then
      call mpi_waitall( int(n_requests), self%requests(1:n_requests),         &
                        mpi_statuses_ignore, ierr )
    end if

    do k = 1, size(self%recv_index)
      data_with_halos(self%recv_index(k)) = self%recv_buffer32(k)
    end do

    n_exchanges = n_exchanges + 1

    ! Bit patterns, for the reasons given in the 64-bit twin: the claim is
    ! bit-identity, and `-finit-real=snan` puts a NaN in every halo word no
    ! exchange has written yet.
    if ( allocated(reference) ) then
      mismatches = 0
      do k = self%n_owned + 1, size(data_with_halos)
        if ( transfer(data_with_halos(k), 0_int32) /=                        &
             transfer(reference(k), 0_int32) ) mismatches = mismatches + 1
      end do
      call halo_plan_count_check( mismatches )
      deallocate( reference )
    end if

  end subroutine halo_plan_exchange_real32

#endif

  !> @brief Reports whether the replay is asked for.
  !> @details Read once, for the reason the shared allocator's own default
  !>          gives: a knob that changed part way through a run would be far
  !>          harder to reason about than one that did not.
  !>
  !>          The replay refuses to run beside `LFRIC_KOKKOS_MEM_ADVICE=host`
  !>          and says so. Under that advice the field pages are kept on the
  !>          host deliberately, so a gather that will later move to the
  !>          device would fault the whole owned block to the card and lose
  !>          more than the gather costs. Refusing loudly is the point: a
  !>          silent fallback would make an A/B row read as a null lever.
  !> @return enabled  Whether the replay path should be taken.
  function halo_plan_mode_enabled() result(enabled)

    implicit none

    logical(l_def) :: enabled

    character(str_def) :: setting
    integer            :: status

    if ( .not. mode_decided ) then
      setting = ''
      call get_environment_variable( mode_variable, setting, status=status )
      if ( status /= 0 ) setting = ''
      mode_on = ( trim(adjustl(setting)) == '1' )

      if ( mode_on ) then
        setting = ''
        call get_environment_variable( advice_variable, setting,             &
                                       status=status )
        if ( status /= 0 ) setting = ''
        if ( trim(adjustl(setting)) == 'host' ) then
          call log_event(                                                    &
              'halo_exchange_plan: ' // mode_variable // '=1 refused '    // &
              'because ' // advice_variable // '=host; the replay gathers '//&
              'from field pages that this advice keeps on the host',         &
              LOG_LEVEL_ERROR )
        end if
      end if

      mode_decided = .true._l_def
    end if

    enabled = mode_on

  end function halo_plan_mode_enabled

  !> @brief Reports whether YAXT should be run beside the replay and compared.
  !> @details The falsifier. With this on, every exchange is done twice -- once
  !>          by YAXT on a copy and once by the replay -- and the halo slabs
  !>          compared word for word. It is far too expensive to measure with
  !>          and is not meant to be; it is what says the extracted lists are
  !>          YAXT's lists, and it is what a first run on new hardware should
  !>          do before it trusts anything here.
  !> @return enabled  Whether to check every exchange against YAXT.
  function halo_plan_selfcheck_enabled() result(enabled)

    implicit none

    logical(l_def) :: enabled

    character(str_def) :: setting
    integer            :: status

    if ( .not. check_decided ) then
      setting = ''
      call get_environment_variable( check_variable, setting, status=status )
      if ( status /= 0 ) setting = ''
      check_on      = ( trim(adjustl(setting)) == '1' )
      check_decided = .true._l_def
    end if

    enabled = check_on

  end function halo_plan_selfcheck_enabled

  !> @brief Counts one exchange the replay could not serve.
  !> @details Called by the caller rather than here, because the reasons a
  !>          routing is not served -- a type this module does not pack, a
  !>          rank with no neighbours -- are the caller's to know.
  subroutine halo_plan_count_fallback()

    implicit none

    n_fallbacks = n_fallbacks + 1

  end subroutine halo_plan_count_fallback

  !> @brief Records the result of one selfcheck comparison.
  !> @param[in] mismatches  How many halo words differed from YAXT's.
  subroutine halo_plan_count_check( mismatches )

    implicit none

    integer(i_def), intent(in) :: mismatches

    n_checked = n_checked + 1
    if ( mismatches > 0 ) n_mismatch = n_mismatch + mismatches

  end subroutine halo_plan_count_check

  !> @brief Writes what the knobs resolved to, and the counts, to standard
  !>        error.
  !> @details Called from the driver's finalise, beside the other Kokkos
  !>          reports and to standard error for the same reason: the model
  !>          finalises the logger before it finalises Kokkos. The line says
  !>          what the knob resolved to whether or not it was set, so that a
  !>          measurement showing no gain can be checked against a knob that
  !>          never came on.
  subroutine halo_plan_report()

    implicit none

    character(3) :: mode
    character(3) :: check

    if ( halo_plan_mode_enabled() ) then
      mode = 'on '
    else
      mode = 'off'
    end if
    if ( halo_plan_selfcheck_enabled() ) then
      check = 'on '
    else
      check = 'off'
    end if

    write( error_unit, '(A,A,A,A,A,I0,A,I0,A,I0,A,I0,A,I0)' )               &
        'lfric_kokkos_halo: mode=', trim(mode),                             &
        ' selfcheck=', trim(check),                                         &
        ' plans=', n_plans,                                                 &
        ' exchanges=', n_exchanges,                                         &
        ' fallbacks=', n_fallbacks,                                         &
        ' checked=', n_checked,                                             &
        ' mismatches=', n_mismatch
    flush( error_unit )

  end subroutine halo_plan_report

end module halo_exchange_plan_mod
