!-----------------------------------------------------------------------------
! (C) Crown copyright 2026 Met Office. All rights reserved.
! The file LICENCE, distributed with this code, contains details of the terms
! under which the code may be used.
!-----------------------------------------------------------------------------

!> @brief Allocates array storage that a Kokkos region can also reach.

!> @details Field data has to be readable both by the Fortran that owns the
!>          field and by a generated Kokkos region that takes an unmanaged
!>          view over it. Allocating it from Kokkos::SharedSpace gives one
!>          address that is valid on both sides, leaving page migration to the
!>          Kokkos runtime rather than to the generated region.
!>
!>          This is a module of its own, rather than code inside field_mod,
!>          because of how the LFRic build preprocesses. field_mod.t90 is
!>          templated to lowercase .f90, and compile.mk has two Fortran rules:
!>          '%.o: %.F90' passes $(MACRO_ARGS) and '%.o: %.f90' does not.
!>          USE_KOKKOS arrives through PRE_PROCESS_MACROS, so it is invisible
!>          inside any templated source, and no template in the tree uses the
!>          preprocessor. Every conditional therefore lives here, in a .F90,
!>          and callers never mention USE_KOKKOS -- the arrangement
!>          driver_kokkos_mod already uses for the runtime itself.
!>
!>          Built without USE_KOKKOS the allocator is a plain ALLOCATE and the
!>          accounting reports zero, so a Fortran-only build behaves exactly as
!>          it did before this module existed.
!>
!>          Even with USE_KOKKOS the allocator falls back to ALLOCATE when the
!>          Kokkos runtime is not running, so one run may hold a mixture of
!>          shared-space arrays and ordinary ones. Which is which is a property
!>          of each array, not of the run, so kokkos_shared_free asks the C++
!>          side whether it issued that particular address rather than
!>          consulting a flag.
!>
!>          kokkos_shared_claim allocates a block and this module owns it
!>          until either kokkos_shared_release gives that one block back or
!>          kokkos_shared_release_all gives every remaining one back on the
!>          way out. The registry is the authority on which blocks are live:
!>          a Fortran object cannot own a block it holds by pointer, because
!>          ALLOCATE with SOURCE= duplicates such an object without
!>          duplicating the storage and without giving the type any way to
!>          notice, so the copy and the original would each free it. The
!>          registry answers "did I issue this block", never "how many
!>          objects point at it", so per-object release is correct exactly
!>          when every object pointing at a block is the one that claimed it
!>          -- the copy rule the design states, and which
!>          n_unknown_releases detects the breach of.
!>
!>          kokkos_shared_live_blocks reports how many are outstanding, so
!>          that a caller can show the count is bounded rather than assert it,
!>          and kokkos_shared_unknown_releases reports how many releases named
!>          a block nothing here had issued.
!>
module kokkos_memory_mod

  ! real32, real64 and int32 are the kinds field_mod.t90 is instantiated for.
  ! They come from iso_fortran_env rather than constants_mod because
  ! constants_mod imports them without exporting them.
  use, intrinsic :: iso_fortran_env, only : real32, real64, int32, error_unit

  use, intrinsic :: iso_c_binding,   only : c_ptr, c_size_t, c_int,          &
                                            c_associated, c_f_pointer,       &
                                            c_loc, c_sizeof

  use constants_mod,                 only : i_def, i_long, l_def, str_def

  implicit none

  private

  public :: kokkos_shared_allocate,           &
            kokkos_shared_free,               &
            kokkos_shared_claim,              &
            kokkos_shared_release,            &
            kokkos_shared_token_issued,       &
            kokkos_shared_note_stale_finalisation, &
            kokkos_shared_release_all,        &
            kokkos_shared_claimed_blocks,     &
            kokkos_shared_live_blocks,        &
            kokkos_shared_unknown_releases,   &
            kokkos_shared_fields_by_default,  &
            kokkos_shared_default_for_setting, &
            kokkos_shared_bytes,              &
            kokkos_shared_peak_bytes,         &
            kokkos_shared_report,             &
            kokkos_shared_report_lifetime,    &
            kokkos_shared_name_refusal,       &
            kokkos_shared_owns,               &
            kokkos_shared_in_use

  !> @brief One block this module has claimed and still owns.
  !> @details Exactly one of the three pointers is associated, which is what
  !>          says the block's kind. The Fortran pointer is held rather than
  !>          the address, because a block that fell back to ALLOCATE has to be
  !>          released by DEALLOCATE of that same pointer; one rebuilt from an
  !>          address by c_f_pointer would not be it.
  type :: shared_block_type
    real(real32),   pointer :: values_real32( : ) => null()
    real(real64),   pointer :: values_real64( : ) => null()
    integer(int32), pointer :: values_int32( : )  => null()
    !> The serial number this block was issued under. A releasing object
    !> presents the number it was given when it claimed, and a block is only
    !> freed when the two agree, so that an object holding a pointer to an
    !> address the registry has since issued to somebody else cannot free
    !> that somebody else's block. Never reused within a run.
    integer(i_def)          :: token = 0_i_def
  end type shared_block_type

  !> Blocks claimed and not yet released. Grown by doubling.
  type(shared_block_type), allocatable, save :: claimed( : )

  !> How many leading entries of 'claimed' are in use.
  integer(i_def), save :: n_claimed = 0_i_def

  !> The largest n_claimed has ever been. Kept because release_all takes
  !> n_claimed back to zero, so the count a caller needs in order to show its
  !> opt-in set is bounded is gone by the time anything reports it.
  integer(i_def), save :: peak_claimed = 0_i_def

  !> Releases of a pointer no registry entry owned. Every block a field
  !> claims is released by that field, so a non-zero count here names a
  !> second object that was pointing at a block a first object had already
  !> given back -- which is to say a shallow copy the copy rule missed. It is
  !> never reset, so the figure the shutdown report writes is the whole run's.
  integer(i_def), save :: n_unknown_releases = 0_i_def

  !> Whether the environment has been consulted for the field default yet.
  logical(l_def), save :: default_decided = .false._l_def

  !> The answer it gave. Only meaningful once default_decided is true.
  logical(l_def), save :: default_shared = .false._l_def

  !> The environment variable that turns the default off at run time.
  character(*), parameter :: shared_fields_variable = 'LFRIC_KOKKOS_SHARED_FIELDS'

  !> Whether the environment has been consulted for the keep-released
  !> diagnostic yet.
  logical(l_def), save :: keep_decided = .false._l_def

  !> The answer it gave. Only meaningful once keep_decided is true.
  logical(l_def), save :: keep_released = .false._l_def

  !> The environment variable that turns the keep-released diagnostic on.
  character(*), parameter :: keep_released_variable = 'LFRIC_KOKKOS_SHARED_KEEP'

  !> Whether the environment has been consulted for the refusal trace yet.
  logical(l_def), save :: trace_decided = .false._l_def

  !> Which refusal the trace stops on. Zero, the default, stops on none.
  !> Only meaningful once trace_decided is true.
  integer(i_def), save :: trace_refusal = 0_i_def

  !> The environment variable that turns the refusal backtrace on, set to
  !> the number of the refusal to trace. A refusal names the field that made
  !> it, which says what was copied; the backtrace says where the copy was
  !> made, and is had by stopping the run, so it is asked for one refusal at
  !> a time rather than given.
  character(*), parameter :: trace_variable = 'LFRIC_KOKKOS_LIFETIME_TRACE'

  !> Finalisations that ran on memory no field had claimed. A field that
  !> claimed carries the serial number it was issued; memory that was never a
  !> field carries whatever was there, which is a number this run has not
  !> issued. Counted apart from n_unknown_releases because it says nothing
  !> about copies: nothing was released, and no block was ever at risk.
  integer(i_def), save :: n_stale_finalisations = 0_i_def

  !> The last serial number issued. A block's token is its value at the
  !> moment it was claimed, so no two blocks of a run share one and a token
  !> read out of memory that is no longer a live field matches nothing.
  integer(i_def), save :: last_token = 0_i_def

  !> Releases refused so far, reported one line each up to this many, so that
  !> a run that meets the case says where without drowning its output.
  integer(i_def), parameter :: refusals_reported = 5_i_def

  !> Entries the registry starts with, and the least it ever grows to.
  integer(i_def), parameter :: initial_registry_size = 16_i_def

  !> @brief Allocates a rank-one pointer array, from shared space where that
  !>        is available and by ALLOCATE where it is not.
  interface kokkos_shared_allocate
    module procedure kokkos_shared_allocate_real32
    module procedure kokkos_shared_allocate_real64
    module procedure kokkos_shared_allocate_int32
  end interface kokkos_shared_allocate

  !> @brief Allocates a rank-one pointer array that this module keeps owning
  !>        until kokkos_shared_release_all.
  interface kokkos_shared_claim
    module procedure kokkos_shared_claim_real32
    module procedure kokkos_shared_claim_real64
    module procedure kokkos_shared_claim_int32
  end interface kokkos_shared_claim

  !> @brief Releases one array that kokkos_shared_claim claimed, giving its
  !>        registry entry back.
  interface kokkos_shared_release
    module procedure kokkos_shared_release_real32
    module procedure kokkos_shared_release_real64
    module procedure kokkos_shared_release_int32
  end interface kokkos_shared_release

  !> @brief Releases an array that kokkos_shared_allocate returned, by
  !>        whichever route it was allocated through.
  interface kokkos_shared_free
    module procedure kokkos_shared_free_real32
    module procedure kokkos_shared_free_real64
    module procedure kokkos_shared_free_int32
  end interface kokkos_shared_free

  !> @brief Reports whether the shared allocator issued the storage an array
  !>        is associated with.
  interface kokkos_shared_owns
    module procedure kokkos_shared_owns_real32
    module procedure kokkos_shared_owns_real64
    module procedure kokkos_shared_owns_int32
  end interface kokkos_shared_owns

#ifdef USE_KOKKOS
  interface

    !> Wraps Kokkos::kokkos_malloc<Kokkos::SharedSpace>. Returns a null
    !> pointer for a zero-byte request or when the runtime is not running.
    function lfric_kokkos_shared_allocate( nbytes )                          &
             bind(c, name='lfric_kokkos_shared_allocate') result(address)
      import :: c_ptr, c_size_t
      implicit none
      integer(c_size_t), value, intent(in) :: nbytes
      type(c_ptr) :: address
    end function lfric_kokkos_shared_allocate

    !> Returns 1 if the allocator issued this address and has freed it, and 0
    !> if it did not, in which case the memory is the caller's to DEALLOCATE.
    function lfric_kokkos_shared_free( address, nbytes )                     &
             bind(c, name='lfric_kokkos_shared_free') result(owned)
      import :: c_ptr, c_size_t, c_int
      implicit none
      type(c_ptr),       value, intent(in) :: address
      integer(c_size_t), value, intent(in) :: nbytes
      integer(c_int) :: owned
    end function lfric_kokkos_shared_free

    function lfric_kokkos_shared_bytes()                                     &
             bind(c, name='lfric_kokkos_shared_bytes') result(bytes)
      import :: c_size_t
      implicit none
      integer(c_size_t) :: bytes
    end function lfric_kokkos_shared_bytes

    function lfric_kokkos_shared_peak_bytes()                                &
             bind(c, name='lfric_kokkos_shared_peak_bytes') result(bytes)
      import :: c_size_t
      implicit none
      integer(c_size_t) :: bytes
    end function lfric_kokkos_shared_peak_bytes

    !> Returns the one-based registry index recorded against an address, or
    !> zero if the allocator did not issue it or no entry owns it.
    function lfric_kokkos_shared_index( address )                            &
             bind(c, name='lfric_kokkos_shared_index') result(index)
      import :: c_ptr, c_size_t
      implicit none
      type(c_ptr), value, intent(in) :: address
      integer(c_size_t) :: index
    end function lfric_kokkos_shared_index

    !> Returns 1 if the allocator issued this address and has not reclaimed
    !> it, and 0 otherwise. Distinct from the index query above, which
    !> answers zero for an address it never issued and for one it issued
    !> without a registry entry owning it alike.
    function lfric_kokkos_shared_owns( address )                             &
             bind(c, name='lfric_kokkos_shared_owns') result(owns)
      import :: c_ptr, c_int
      implicit none
      type(c_ptr), value, intent(in) :: address
      integer(c_int) :: owns
    end function lfric_kokkos_shared_owns

    !> Records the one-based registry index owning an address. Does nothing
    !> if the allocator did not issue that address.
    subroutine lfric_kokkos_shared_set_index( address, index )               &
               bind(c, name='lfric_kokkos_shared_set_index')
      import :: c_ptr, c_size_t
      implicit none
      type(c_ptr),       value, intent(in) :: address
      integer(c_size_t), value, intent(in) :: index
    end subroutine lfric_kokkos_shared_set_index

    subroutine lfric_kokkos_shared_report()                                  &
               bind(c, name='lfric_kokkos_shared_report')
    end subroutine lfric_kokkos_shared_report

  end interface
#endif

contains

  !> @brief Allocates a 32-bit real array, from shared space where available.
  !> @details The pointer must have a defined association status on entry;
  !>          field_mod's components are default-initialised to null for that
  !>          reason. An already associated pointer is released first, so the
  !>          call cannot leak.
  !> @param [in,out] array   Pointer to allocate. On return it is associated
  !>                         with 'length' elements, lower bound one.
  !> @param [in] length      Number of elements required.
  subroutine kokkos_shared_allocate_real32( array, length )

    implicit none

    real(real32), pointer, intent(inout) :: array( : )
    integer(i_def),        intent(in)    :: length

#ifdef USE_KOKKOS
    real(real32)      :: element = 0.0_real32
    integer(c_size_t) :: nbytes
    type(c_ptr)       :: address
#endif

    if ( associated(array) ) call kokkos_shared_free( array )

#ifdef USE_KOKKOS
    nbytes = int(length, c_size_t) * c_sizeof(element)
    address = lfric_kokkos_shared_allocate( nbytes )
    if ( c_associated(address) ) then
      call c_f_pointer( address, array, [length] )
      return
    end if
#endif

    allocate( array(length) )

  end subroutine kokkos_shared_allocate_real32

  !> @brief Allocates a 64-bit real array, from shared space where available.
  !> @details The pointer must have a defined association status on entry;
  !>          field_mod's components are default-initialised to null for that
  !>          reason. An already associated pointer is released first, so the
  !>          call cannot leak.
  !> @param [in,out] array   Pointer to allocate. On return it is associated
  !>                         with 'length' elements, lower bound one.
  !> @param [in] length      Number of elements required.
  subroutine kokkos_shared_allocate_real64( array, length )

    implicit none

    real(real64), pointer, intent(inout) :: array( : )
    integer(i_def),        intent(in)    :: length

#ifdef USE_KOKKOS
    real(real64)      :: element = 0.0_real64
    integer(c_size_t) :: nbytes
    type(c_ptr)       :: address
#endif

    if ( associated(array) ) call kokkos_shared_free( array )

#ifdef USE_KOKKOS
    nbytes = int(length, c_size_t) * c_sizeof(element)
    address = lfric_kokkos_shared_allocate( nbytes )
    if ( c_associated(address) ) then
      call c_f_pointer( address, array, [length] )
      return
    end if
#endif

    allocate( array(length) )

  end subroutine kokkos_shared_allocate_real64

  !> @brief Allocates a 32-bit integer array, from shared space where
  !>        available.
  !> @details The pointer must have a defined association status on entry;
  !>          field_mod's components are default-initialised to null for that
  !>          reason. An already associated pointer is released first, so the
  !>          call cannot leak.
  !> @param [in,out] array   Pointer to allocate. On return it is associated
  !>                         with 'length' elements, lower bound one.
  !> @param [in] length      Number of elements required.
  subroutine kokkos_shared_allocate_int32( array, length )

    implicit none

    integer(int32), pointer, intent(inout) :: array( : )
    integer(i_def),          intent(in)    :: length

#ifdef USE_KOKKOS
    integer(int32)    :: element = 0_int32
    integer(c_size_t) :: nbytes
    type(c_ptr)       :: address
#endif

    if ( associated(array) ) call kokkos_shared_free( array )

#ifdef USE_KOKKOS
    nbytes = int(length, c_size_t) * c_sizeof(element)
    address = lfric_kokkos_shared_allocate( nbytes )
    if ( c_associated(address) ) then
      call c_f_pointer( address, array, [length] )
      return
    end if
#endif

    allocate( array(length) )

  end subroutine kokkos_shared_allocate_int32

  !> @brief Releases a 32-bit real array allocated by this module.
  !> @details Asks the C++ side whether it issued this address rather than
  !>          assuming, because a run may hold both shared-space and ordinary
  !>          arrays and freeing either of them by the other route is memory
  !>          corruption. A zero-sized array has no first element to take the
  !>          address of, and can only have come from ALLOCATE, so it takes
  !>          the ordinary route without asking.
  !> @param [in,out] array   Pointer to release. Does nothing if it is not
  !>                         associated. On return it is disassociated.
  subroutine kokkos_shared_free_real32( array )

    implicit none

    real(real32), pointer, intent(inout) :: array( : )

#ifdef USE_KOKKOS
    real(real32)      :: element = 0.0_real32
    integer(c_size_t) :: nbytes
#endif

    if ( .not. associated(array) ) return

#ifdef USE_KOKKOS
    if ( size(array) > 0 ) then
      nbytes = int(size(array), c_size_t) * c_sizeof(element)
      if ( lfric_kokkos_shared_free( c_loc(array(1)), nbytes )               &
           /= 0_c_int ) then
        nullify( array )
        return
      end if
    end if
#endif

    deallocate( array )

  end subroutine kokkos_shared_free_real32

  !> @brief Releases a 64-bit real array allocated by this module.
  !> @details Asks the C++ side whether it issued this address rather than
  !>          assuming, because a run may hold both shared-space and ordinary
  !>          arrays and freeing either of them by the other route is memory
  !>          corruption. A zero-sized array has no first element to take the
  !>          address of, and can only have come from ALLOCATE, so it takes
  !>          the ordinary route without asking.
  !> @param [in,out] array   Pointer to release. Does nothing if it is not
  !>                         associated. On return it is disassociated.
  subroutine kokkos_shared_free_real64( array )

    implicit none

    real(real64), pointer, intent(inout) :: array( : )

#ifdef USE_KOKKOS
    real(real64)      :: element = 0.0_real64
    integer(c_size_t) :: nbytes
#endif

    if ( .not. associated(array) ) return

#ifdef USE_KOKKOS
    if ( size(array) > 0 ) then
      nbytes = int(size(array), c_size_t) * c_sizeof(element)
      if ( lfric_kokkos_shared_free( c_loc(array(1)), nbytes )               &
           /= 0_c_int ) then
        nullify( array )
        return
      end if
    end if
#endif

    deallocate( array )

  end subroutine kokkos_shared_free_real64

  !> @brief Releases a 32-bit integer array allocated by this module.
  !> @details Asks the C++ side whether it issued this address rather than
  !>          assuming, because a run may hold both shared-space and ordinary
  !>          arrays and freeing either of them by the other route is memory
  !>          corruption. A zero-sized array has no first element to take the
  !>          address of, and can only have come from ALLOCATE, so it takes
  !>          the ordinary route without asking.
  !> @param [in,out] array   Pointer to release. Does nothing if it is not
  !>                         associated. On return it is disassociated.
  subroutine kokkos_shared_free_int32( array )

    implicit none

    integer(int32), pointer, intent(inout) :: array( : )

#ifdef USE_KOKKOS
    integer(int32)    :: element = 0_int32
    integer(c_size_t) :: nbytes
#endif

    if ( .not. associated(array) ) return

#ifdef USE_KOKKOS
    if ( size(array) > 0 ) then
      nbytes = int(size(array), c_size_t) * c_sizeof(element)
      if ( lfric_kokkos_shared_free( c_loc(array(1)), nbytes )               &
           /= 0_c_int ) then
        nullify( array )
        return
      end if
    end if
#endif

    deallocate( array )

  end subroutine kokkos_shared_free_int32

  !> @brief Records a claimed block, growing the registry when it is full.
  !> @param [in] record   The block to take ownership of. Exactly one of its
  !>                      three pointers is expected to be associated.
  subroutine remember_block( record, token )

    implicit none

    type(shared_block_type), intent(in)  :: record
    integer(i_def),          intent(out) :: token

    type(shared_block_type), allocatable :: bigger( : )

    if ( .not. allocated(claimed) ) then
      allocate( claimed(initial_registry_size) )
      n_claimed = 0_i_def
    end if

    if ( n_claimed == size(claimed, kind=i_def) ) then
      allocate( bigger(2_i_def * n_claimed) )
      bigger(1:n_claimed) = claimed(1:n_claimed)
      call move_alloc( bigger, claimed )
    end if

    n_claimed = n_claimed + 1_i_def
    claimed(n_claimed) = record
    last_token = last_token + 1_i_def
    claimed(n_claimed)%token = last_token
    token = last_token
    call note_index( claimed(n_claimed), n_claimed )

    if ( n_claimed > peak_claimed ) peak_claimed = n_claimed

  end subroutine remember_block

  !> @brief Tells the allocator which registry entry owns a block.
  !> @details Only the shared allocator keeps this index, so a block that fell
  !>          back to ALLOCATE is silently not recorded and is found later by
  !>          the scan in find_slot_* instead. Without USE_KOKKOS there is no
  !>          allocator to tell and this does nothing at all.
  !> @param [in] record   The registry entry, with one pointer associated.
  !> @param [in] index    Its one-based position in 'claimed'.
  subroutine note_index( record, index )

    implicit none

    type(shared_block_type), intent(in) :: record
    integer(i_def),          intent(in) :: index

#ifdef USE_KOKKOS
    if ( associated(record%values_real32) ) then
      if ( size(record%values_real32) > 0 )                                  &
        call lfric_kokkos_shared_set_index( c_loc(record%values_real32(1)),  &
                                            int(index, c_size_t) )
    else if ( associated(record%values_real64) ) then
      if ( size(record%values_real64) > 0 )                                  &
        call lfric_kokkos_shared_set_index( c_loc(record%values_real64(1)),  &
                                            int(index, c_size_t) )
    else if ( associated(record%values_int32) ) then
      if ( size(record%values_int32) > 0 )                                   &
        call lfric_kokkos_shared_set_index( c_loc(record%values_int32(1)),   &
                                            int(index, c_size_t) )
    end if
#else
    ! Without the shared allocator there is no map to maintain, and the
    ! registry is found by the scan in find_slot_* instead.
#endif

  end subroutine note_index

  !> @brief Removes a registry entry, keeping the used entries contiguous.
  !> @details The last entry is moved into the hole rather than the tail being
  !>          shuffled down, so a release costs the same whichever entry it
  !>          names. The moved entry's index changes, so the allocator is told
  !>          its new one; the vacated entry is nullified so that a stale
  !>          pointer cannot be freed a second time by release_all.
  !> @param [in] slot   One-based index of the entry to remove.
  subroutine forget_block( slot )

    implicit none

    integer(i_def), intent(in) :: slot

    if ( slot /= n_claimed ) then
      claimed(slot) = claimed(n_claimed)
      call note_index( claimed(slot), slot )
    end if

    nullify( claimed(n_claimed)%values_real32,                               &
             claimed(n_claimed)%values_real64,                               &
             claimed(n_claimed)%values_int32 )
    claimed(n_claimed)%token = 0_i_def

    n_claimed = n_claimed - 1_i_def

  end subroutine forget_block

  !> @brief Claims a 32-bit real array on the caller's behalf.
  !> @details The pointer is disassociated before allocating rather than being
  !>          handed straight to kokkos_shared_allocate, which releases an
  !>          associated pointer on entry. A claimed block belongs to the
  !>          registry, so a caller passing one back in must not have it freed
  !>          from under the registry's record of it.
  !> @param [in,out] array   Pointer to allocate. On return it is associated
  !>                         with 'length' elements, lower bound one.
  !> @param [in] length      Number of elements required.
  !> @param [out] token      The serial number this claim was issued under,
  !>                         to be presented again when the block is released.
  !>                         Zero for a zero-length block, which the registry
  !>                         does not record. Optional so that a caller with
  !>                         no lifetime of its own to prove need not hold one.
  subroutine kokkos_shared_claim_real32( array, length, token )

    implicit none

    real(real32), pointer, intent(inout) :: array( : )
    integer(i_def),        intent(in)    :: length
    integer(i_def),        optional, intent(out) :: token

    type(shared_block_type) :: record

    integer(i_def) :: issued

    array => null()
    call kokkos_shared_allocate( array, length )

    ! A zero-length block is never registered. ASSOCIATED(p, t) is false for
    ! zero-sized arrays (F2018 16.9.16), so an entry made for one could not be
    ! found again by the pointer that owns it; its release takes the ordinary
    ! DEALLOCATE route instead, which is where it came from -- the shared
    ! allocator returns a null pointer for a zero-byte request.
    if ( size(array) == 0 ) then
      if ( present(token) ) token = 0_i_def
      return
    end if

    record%values_real32 => array
    call remember_block( record, issued )
    if ( present(token) ) token = issued

  end subroutine kokkos_shared_claim_real32

  !> @brief Claims a 64-bit real array on the caller's behalf.
  !> @details The pointer is disassociated before allocating rather than being
  !>          handed straight to kokkos_shared_allocate, which releases an
  !>          associated pointer on entry. A claimed block belongs to the
  !>          registry, so a caller passing one back in must not have it freed
  !>          from under the registry's record of it.
  !> @param [in,out] array   Pointer to allocate. On return it is associated
  !>                         with 'length' elements, lower bound one.
  !> @param [in] length      Number of elements required.
  !> @param [out] token      The serial number this claim was issued under,
  !>                         to be presented again when the block is released.
  !>                         Zero for a zero-length block, which the registry
  !>                         does not record. Optional so that a caller with
  !>                         no lifetime of its own to prove need not hold one.
  subroutine kokkos_shared_claim_real64( array, length, token )

    implicit none

    real(real64), pointer, intent(inout) :: array( : )
    integer(i_def),        intent(in)    :: length
    integer(i_def),        optional, intent(out) :: token

    type(shared_block_type) :: record

    integer(i_def) :: issued

    array => null()
    call kokkos_shared_allocate( array, length )

    ! A zero-length block is never registered. ASSOCIATED(p, t) is false for
    ! zero-sized arrays (F2018 16.9.16), so an entry made for one could not be
    ! found again by the pointer that owns it; its release takes the ordinary
    ! DEALLOCATE route instead, which is where it came from -- the shared
    ! allocator returns a null pointer for a zero-byte request.
    if ( size(array) == 0 ) then
      if ( present(token) ) token = 0_i_def
      return
    end if

    record%values_real64 => array
    call remember_block( record, issued )
    if ( present(token) ) token = issued

  end subroutine kokkos_shared_claim_real64

  !> @brief Claims a 32-bit integer array on the caller's behalf.
  !> @details The pointer is disassociated before allocating rather than being
  !>          handed straight to kokkos_shared_allocate, which releases an
  !>          associated pointer on entry. A claimed block belongs to the
  !>          registry, so a caller passing one back in must not have it freed
  !>          from under the registry's record of it.
  !> @param [in,out] array   Pointer to allocate. On return it is associated
  !>                         with 'length' elements, lower bound one.
  !> @param [in] length      Number of elements required.
  !> @param [out] token      The serial number this claim was issued under,
  !>                         to be presented again when the block is released.
  !>                         Zero for a zero-length block, which the registry
  !>                         does not record. Optional so that a caller with
  !>                         no lifetime of its own to prove need not hold one.
  subroutine kokkos_shared_claim_int32( array, length, token )

    implicit none

    integer(int32), pointer, intent(inout) :: array( : )
    integer(i_def),          intent(in)    :: length
    integer(i_def),        optional, intent(out) :: token

    type(shared_block_type) :: record

    integer(i_def) :: issued

    array => null()
    call kokkos_shared_allocate( array, length )

    ! A zero-length block is never registered. ASSOCIATED(p, t) is false for
    ! zero-sized arrays (F2018 16.9.16), so an entry made for one could not be
    ! found again by the pointer that owns it; its release takes the ordinary
    ! DEALLOCATE route instead, which is where it came from -- the shared
    ! allocator returns a null pointer for a zero-byte request.
    if ( size(array) == 0 ) then
      if ( present(token) ) token = 0_i_def
      return
    end if

    record%values_int32 => array
    call remember_block( record, issued )
    if ( present(token) ) token = issued

  end subroutine kokkos_shared_claim_int32

  !> @brief Says whether a token is one this run issued.
  !> @details The registry's serial numbers start at one and are never
  !>          reused, so a value outside the range it has issued cannot have
  !>          come from a claim. That is what lets an object about to release
  !>          ask whether it is a field at all: a finaliser runs on whatever
  !>          memory it is pointed at, and a type holding a field by value in
  !>          memory that was never initialised reads a claimed flag and a
  !>          token out of what happened to be there. Such a caller is told
  !>          no here and does not reach the registry, so that a run's
  !>          unknown-release count stays what it is for -- the copy detector
  !>          of the field-lifetime design -- rather than counting garbage.
  !>          A copy of a live field carries a token this run did issue and
  !>          is still refused, and counted, by the registry.
  !> @param [in] token   The token to judge.
  !> @return issued      Whether the registry issued it.
  function kokkos_shared_token_issued( token ) result(issued)

    implicit none

    integer(i_def), intent(in) :: token
    logical(l_def) :: issued

    issued = ( token > 0_i_def .and. token <= last_token )

  end function kokkos_shared_token_issued

  !> @brief Counts a finalisation that ran on memory no field had claimed.
  !> @details Reported at shutdown, on a line of its own and only when there
  !>          were any, because it is a fact about the model's finalisation
  !>          and not about this module's book-keeping.
  subroutine kokkos_shared_note_stale_finalisation()

    implicit none

    n_stale_finalisations = n_stale_finalisations + 1_i_def

  end subroutine kokkos_shared_note_stale_finalisation

  !> @brief Records a release this module would not honour.
  !> @details A release names a block by its pointer and its token. Either
  !>          can fail to match: a pointer no entry holds, or a pointer whose
  !>          entry was issued under another token. Both mean the same thing
  !>          -- the caller was not the owner of what it asked to give back --
  !>          and both are answered the same way, by leaving the block alone
  !>          and counting the attempt. Freeing it would be the second free of
  !>          a block that is either gone already or somebody else's.
  !>
  !>          The first few are reported as they happen, because a count at
  !>          shutdown says how many and not where, and the line is written to
  !>          standard error for the same reason the shutdown report is: the
  !>          logger is finalised before Kokkos is.
  subroutine refuse_release( reason )

    implicit none

    character(*), optional, intent(in) :: reason

    character(str_def) :: told

    n_unknown_releases = n_unknown_releases + 1_i_def

    told = 'no entry held the pointer'
    if ( present(reason) ) told = reason

    if ( n_unknown_releases <= refusals_reported ) then
      write( error_unit, '(A,I0,3A)' )                                       &
          'lfric_kokkos_lifetime: refused release ', n_unknown_releases,     &
          ' of a block this registry did not issue (', trim(told), ')'
      flush( error_unit )
    end if

  end subroutine refuse_release

  !> @brief Names the object whose release was just refused.
  !> @details refuse_release counts and reports a refusal without knowing who
  !>          made it, because the registry is handed a pointer and not an
  !>          owner. A caller that knows its own name says so here, on the
  !>          line after, so that a refusal names the field it came from --
  !>          which is what says which copy the copy rule missed. Reported for
  !>          the same first few refusals the registry reports, so that a run
  !>          meeting the case many times does not drown its own output.
  !>
  !>          With LFRIC_KOKKOS_LIFETIME_TRACE=<n> the n-th refusal stops the
  !>          run by ERROR STOP, which gfortran answers with a backtrace of
  !>          the frames that reached it -- the copy's site. gfortran's
  !>          BACKTRACE intrinsic would print the same without stopping and is
  !>          not available: this module is compiled -std=f2008, under which
  !>          the extension is an undefined external. Stopping is the price of
  !>          a standard-conforming trace, which is why it is asked for rather
  !>          than given: it is how a refusal is traced to its site, and the
  !>          next reader of one will want the same route.
  !> @param [in] owner   The name of the object that was refused.
  subroutine kokkos_shared_name_refusal( owner )

    implicit none

    character(*), intent(in) :: owner

    if ( n_unknown_releases > refusals_reported ) return

    write( error_unit, '(2A)' )                                              &
        'lfric_kokkos_lifetime:   the block was held by ', trim(owner)
    flush( error_unit )

    if ( refusal_trace_enabled() == n_unknown_releases ) then
      error stop 'lfric_kokkos_lifetime: refused release, traced on request'
    end if

  end subroutine kokkos_shared_name_refusal

  !> @brief Says which refused release should stop the run for a backtrace.
  !> @details None unless LFRIC_KOKKOS_LIFETIME_TRACE names one by number:
  !>          '1' traces the first refusal, '2' the second, and so on, since
  !>          stopping is how the trace is had and a run has only one stop in
  !>          it. The environment is read once, for the reason the field
  !>          default gives.
  !> @return traced   The number of the refusal to trace, or zero for none.
  function refusal_trace_enabled() result(traced)

    implicit none

    integer(i_def) :: traced

    character(str_def) :: setting
    integer            :: status

    if ( .not. trace_decided ) then
      setting = ''
      call get_environment_variable( trace_variable, setting, status=status )
      if ( status /= 0 ) setting = ''
      trace_refusal = 0_i_def
      if ( len_trim(setting) > 0 ) then
        read( setting, *, iostat=status ) trace_refusal
        if ( status /= 0 ) trace_refusal = 0_i_def
      end if
      trace_decided = .true._l_def
    end if

    traced = trace_refusal

  end function refusal_trace_enabled

  !> @brief Finds the registry entry owning a 32-bit real block.
  !> @details The shared allocator keeps an address-to-index map, so a block
  !>          it issued is found in constant time; a block that fell back to
  !>          Fortran ALLOCATE is not in that map and is found by scanning the
  !>          registry instead. Those are the handful claimed before the
  !>          Kokkos runtime is up, not the run's fields, so the scan is not
  !>          on the hot path -- and in a build without USE_KOKKOS it is the
  !>          only path, which is the build where no field is shared anyway.
  !> @param [in] array   A block to look for. Must be associated and non-empty.
  !> @return slot        Its one-based index in 'claimed', or zero if the
  !>                     registry does not own it.
  function find_slot_real32( array ) result(slot)

    implicit none

    real(real32), pointer, intent(in) :: array( : )
    integer(i_def) :: slot

    integer(i_def) :: i

#ifdef USE_KOKKOS
    slot = int( lfric_kokkos_shared_index( c_loc(array(1)) ), i_def )
    if ( slot > 0_i_def ) return
#endif

    slot = 0_i_def
    if ( .not. allocated(claimed) ) return

    do i = 1_i_def, n_claimed
      if ( associated(claimed(i)%values_real32, array) ) then
        slot = i
        return
      end if
    end do

  end function find_slot_real32

  !> @brief Releases one 32-bit real block back to the registry.
  !> @details This is how a field gives its data back when the field goes
  !>          away, so that the number of live blocks follows the number of
  !>          live fields rather than the number ever created. A pointer the
  !>          registry never issued is left exactly as it was and counted, as
  !>          the design's detector of an unsanctioned copy: it means some
  !>          object was still pointing at a block another object had already
  !>          released, and freeing it here would be the second free.
  !> @param [in,out] array   The block to release. On return it is
  !>                         disassociated if the release happened, and
  !>                         untouched if it did not.
  !> @param [in] token       The serial number the claim returned. A
  !>                         release that presents the wrong one is
  !>                         refused. Optional so that a caller which
  !>                         never held a token -- a test releasing what
  !>                         it claimed in the same scope -- is unchanged.
  !> @return released        Whether this call gave the block back.
  function kokkos_shared_release_real32( array, token ) result(released)

    implicit none

    real(real32), pointer, intent(inout) :: array( : )
    integer(i_def), optional, intent(in) :: token
    logical(l_def) :: released

    integer(i_def) :: slot
    integer(i_def) :: presented

    released = .false._l_def

    presented = 0_i_def
    if ( present(token) ) presented = token

    if ( .not. associated(array) ) return

    ! Not registered when claimed, so not looked for here. A caller holding a
    ! token cannot be the owner of a zero-length block, since one is never
    ! given a token, and so is refused rather than deallocated: the pointer is
    ! then not one this module ever issued.
    if ( size(array) == 0 ) then
      if ( present(token) ) then
        call refuse_release()
        return
      end if
      deallocate( array )
      released = .true._l_def
      return
    end if

    slot = find_slot_real32( array )

    if ( slot == 0_i_def ) then
      call refuse_release()
      return
    end if

    ! The address matched an entry, which is not on its own enough: an object
    ! reading a pointer out of memory that is no longer a live field can name
    ! an address the registry has since issued to another field. The token
    ! says which claim this caller is releasing, and only the claim that is
    ! still live carries it.
    if ( present(token) ) then
      if ( claimed(slot)%token /= presented ) then
        call refuse_release( 'the address is issued under another token' )
        return
      end if
    end if

    ! Freed through the registry's own pointer rather than through 'array',
    ! because a block that fell back to ALLOCATE must be given back by
    ! DEALLOCATE of the pointer that allocated it.
    ! Diagnostic: with LFRIC_KOKKOS_SHARED_KEEP set the entry is removed but
    ! the block is not given back, so a second object still pointing at it
    ! keeps reading live memory. It is the only way to tell a release that
    ! came too early from a write past the end of a block, which look the
    ! same once the heap is corrupt.
    if ( .not. keep_released_blocks() ) then
      call kokkos_shared_free( claimed(slot)%values_real32 )
    end if
    call forget_block( slot )

    nullify( array )
    released = .true._l_def

  end function kokkos_shared_release_real32

  !> @brief Finds the registry entry owning a 64-bit real block.
  !> @details The shared allocator keeps an address-to-index map, so a block
  !>          it issued is found in constant time; a block that fell back to
  !>          Fortran ALLOCATE is not in that map and is found by scanning the
  !>          registry instead. Those are the handful claimed before the
  !>          Kokkos runtime is up, not the run's fields, so the scan is not
  !>          on the hot path -- and in a build without USE_KOKKOS it is the
  !>          only path, which is the build where no field is shared anyway.
  !> @param [in] array   A block to look for. Must be associated and non-empty.
  !> @return slot        Its one-based index in 'claimed', or zero if the
  !>                     registry does not own it.
  function find_slot_real64( array ) result(slot)

    implicit none

    real(real64), pointer, intent(in) :: array( : )
    integer(i_def) :: slot

    integer(i_def) :: i

#ifdef USE_KOKKOS
    slot = int( lfric_kokkos_shared_index( c_loc(array(1)) ), i_def )
    if ( slot > 0_i_def ) return
#endif

    slot = 0_i_def
    if ( .not. allocated(claimed) ) return

    do i = 1_i_def, n_claimed
      if ( associated(claimed(i)%values_real64, array) ) then
        slot = i
        return
      end if
    end do

  end function find_slot_real64

  !> @brief Releases one 64-bit real block back to the registry.
  !> @details This is how a field gives its data back when the field goes
  !>          away, so that the number of live blocks follows the number of
  !>          live fields rather than the number ever created. A pointer the
  !>          registry never issued is left exactly as it was and counted, as
  !>          the design's detector of an unsanctioned copy: it means some
  !>          object was still pointing at a block another object had already
  !>          released, and freeing it here would be the second free.
  !> @param [in,out] array   The block to release. On return it is
  !>                         disassociated if the release happened, and
  !>                         untouched if it did not.
  !> @param [in] token       The serial number the claim returned. A
  !>                         release that presents the wrong one is
  !>                         refused. Optional so that a caller which
  !>                         never held a token -- a test releasing what
  !>                         it claimed in the same scope -- is unchanged.
  !> @return released        Whether this call gave the block back.
  function kokkos_shared_release_real64( array, token ) result(released)

    implicit none

    real(real64), pointer, intent(inout) :: array( : )
    integer(i_def), optional, intent(in) :: token
    logical(l_def) :: released

    integer(i_def) :: slot
    integer(i_def) :: presented

    released = .false._l_def

    presented = 0_i_def
    if ( present(token) ) presented = token

    if ( .not. associated(array) ) return

    ! Not registered when claimed, so not looked for here. A caller holding a
    ! token cannot be the owner of a zero-length block, since one is never
    ! given a token, and so is refused rather than deallocated: the pointer is
    ! then not one this module ever issued.
    if ( size(array) == 0 ) then
      if ( present(token) ) then
        call refuse_release()
        return
      end if
      deallocate( array )
      released = .true._l_def
      return
    end if

    slot = find_slot_real64( array )

    if ( slot == 0_i_def ) then
      call refuse_release()
      return
    end if

    ! The address matched an entry, which is not on its own enough: an object
    ! reading a pointer out of memory that is no longer a live field can name
    ! an address the registry has since issued to another field. The token
    ! says which claim this caller is releasing, and only the claim that is
    ! still live carries it.
    if ( present(token) ) then
      if ( claimed(slot)%token /= presented ) then
        call refuse_release( 'the address is issued under another token' )
        return
      end if
    end if

    ! Freed through the registry's own pointer rather than through 'array',
    ! because a block that fell back to ALLOCATE must be given back by
    ! DEALLOCATE of the pointer that allocated it.
    ! Diagnostic: with LFRIC_KOKKOS_SHARED_KEEP set the entry is removed but
    ! the block is not given back, so a second object still pointing at it
    ! keeps reading live memory. It is the only way to tell a release that
    ! came too early from a write past the end of a block, which look the
    ! same once the heap is corrupt.
    if ( .not. keep_released_blocks() ) then
      call kokkos_shared_free( claimed(slot)%values_real64 )
    end if
    call forget_block( slot )

    nullify( array )
    released = .true._l_def

  end function kokkos_shared_release_real64

  !> @brief Finds the registry entry owning a 32-bit integer block.
  !> @details The shared allocator keeps an address-to-index map, so a block
  !>          it issued is found in constant time; a block that fell back to
  !>          Fortran ALLOCATE is not in that map and is found by scanning the
  !>          registry instead. Those are the handful claimed before the
  !>          Kokkos runtime is up, not the run's fields, so the scan is not
  !>          on the hot path -- and in a build without USE_KOKKOS it is the
  !>          only path, which is the build where no field is shared anyway.
  !> @param [in] array   A block to look for. Must be associated and non-empty.
  !> @return slot        Its one-based index in 'claimed', or zero if the
  !>                     registry does not own it.
  function find_slot_int32( array ) result(slot)

    implicit none

    integer(int32), pointer, intent(in) :: array( : )
    integer(i_def) :: slot

    integer(i_def) :: i

#ifdef USE_KOKKOS
    slot = int( lfric_kokkos_shared_index( c_loc(array(1)) ), i_def )
    if ( slot > 0_i_def ) return
#endif

    slot = 0_i_def
    if ( .not. allocated(claimed) ) return

    do i = 1_i_def, n_claimed
      if ( associated(claimed(i)%values_int32, array) ) then
        slot = i
        return
      end if
    end do

  end function find_slot_int32

  !> @brief Releases one 32-bit integer block back to the registry.
  !> @details This is how a field gives its data back when the field goes
  !>          away, so that the number of live blocks follows the number of
  !>          live fields rather than the number ever created. A pointer the
  !>          registry never issued is left exactly as it was and counted, as
  !>          the design's detector of an unsanctioned copy: it means some
  !>          object was still pointing at a block another object had already
  !>          released, and freeing it here would be the second free.
  !> @param [in,out] array   The block to release. On return it is
  !>                         disassociated if the release happened, and
  !>                         untouched if it did not.
  !> @param [in] token       The serial number the claim returned. A
  !>                         release that presents the wrong one is
  !>                         refused. Optional so that a caller which
  !>                         never held a token -- a test releasing what
  !>                         it claimed in the same scope -- is unchanged.
  !> @return released        Whether this call gave the block back.
  function kokkos_shared_release_int32( array, token ) result(released)

    implicit none

    integer(int32), pointer, intent(inout) :: array( : )
    integer(i_def), optional, intent(in) :: token
    logical(l_def) :: released

    integer(i_def) :: slot
    integer(i_def) :: presented

    released = .false._l_def

    presented = 0_i_def
    if ( present(token) ) presented = token

    if ( .not. associated(array) ) return

    ! Not registered when claimed, so not looked for here. A caller holding a
    ! token cannot be the owner of a zero-length block, since one is never
    ! given a token, and so is refused rather than deallocated: the pointer is
    ! then not one this module ever issued.
    if ( size(array) == 0 ) then
      if ( present(token) ) then
        call refuse_release()
        return
      end if
      deallocate( array )
      released = .true._l_def
      return
    end if

    slot = find_slot_int32( array )

    if ( slot == 0_i_def ) then
      call refuse_release()
      return
    end if

    ! The address matched an entry, which is not on its own enough: an object
    ! reading a pointer out of memory that is no longer a live field can name
    ! an address the registry has since issued to another field. The token
    ! says which claim this caller is releasing, and only the claim that is
    ! still live carries it.
    if ( present(token) ) then
      if ( claimed(slot)%token /= presented ) then
        call refuse_release( 'the address is issued under another token' )
        return
      end if
    end if

    ! Freed through the registry's own pointer rather than through 'array',
    ! because a block that fell back to ALLOCATE must be given back by
    ! DEALLOCATE of the pointer that allocated it.
    ! Diagnostic: with LFRIC_KOKKOS_SHARED_KEEP set the entry is removed but
    ! the block is not given back, so a second object still pointing at it
    ! keeps reading live memory. It is the only way to tell a release that
    ! came too early from a write past the end of a block, which look the
    ! same once the heap is corrupt.
    if ( .not. keep_released_blocks() ) then
      call kokkos_shared_free( claimed(slot)%values_int32 )
    end if
    call forget_block( slot )

    nullify( array )
    released = .true._l_def

  end function kokkos_shared_release_int32

  !> @brief Releases every block this module has claimed.
  !> @details Called on the way out, before Kokkos::finalize, since a block
  !>          from shared space cannot be returned once the runtime has gone.
  !>          Whatever still points at a released block is dangling afterwards,
  !>          which is why this is a shutdown operation and not a way of
  !>          reclaiming storage during a run.
  subroutine kokkos_shared_release_all()

    implicit none

    integer(i_def) :: i

    if ( .not. allocated(claimed) ) return

    do i = n_claimed, 1_i_def, -1_i_def
      if ( associated(claimed(i)%values_real32) ) then
        call kokkos_shared_free( claimed(i)%values_real32 )
      else if ( associated(claimed(i)%values_real64) ) then
        call kokkos_shared_free( claimed(i)%values_real64 )
      else if ( associated(claimed(i)%values_int32) ) then
        call kokkos_shared_free( claimed(i)%values_int32 )
      end if
    end do

    n_claimed = 0_i_def

  end subroutine kokkos_shared_release_all

  !> @brief Reports how many claimed blocks are outstanding.
  !> @details A caller that opts fields in to shared storage can watch this
  !>          across two runs of different length to show that its opt-in set
  !>          is bounded, which the registry itself cannot know.
  !> @return blocks   Blocks claimed and not yet released.
  function kokkos_shared_claimed_blocks() result(blocks)

    implicit none

    integer(i_def) :: blocks

    blocks = n_claimed

  end function kokkos_shared_claimed_blocks

  !> @brief Reports how many claimed blocks are live.
  !> @details The same number as kokkos_shared_claimed_blocks, under the name
  !>          the lifetime gates read. Both are kept: the older name says
  !>          "claimed and not yet released", which was the whole of the story
  !>          when nothing was released before shutdown, and this one says what
  !>          the figure now means.
  !> @return blocks   Blocks claimed and not yet released.
  function kokkos_shared_live_blocks() result(blocks)

    implicit none

    integer(i_def) :: blocks

    blocks = n_claimed

  end function kokkos_shared_live_blocks

  !> @brief Reports how many releases named a block the registry had not
  !>        issued.
  !> @details Expected to be zero. Any other value says an object was
  !>          pointing at a block it had not claimed -- a copy made by
  !>          ALLOCATE with SOURCE= rather than through initialise -- and the
  !>          block it released was released a second time.
  !> @return releases   Unknown releases so far in this run.
  function kokkos_shared_unknown_releases() result(releases)

    implicit none

    integer(i_def) :: releases

    releases = n_unknown_releases

  end function kokkos_shared_unknown_releases

  !> @brief Decides whether fields take their data from shared space, given
  !>        the setting of the environment variable that can turn it off.
  !> @details Separated from kokkos_shared_fields_by_default so that the
  !>          decision can be tested: a Fortran unit test cannot portably set
  !>          an environment variable in the process it is running in. The
  !>          build decides the answer and the variable can only say no, so
  !>          that a run without it set behaves as the build was compiled to.
  !> @param [in] setting   The variable's value, blank when it is not set.
  !> @return shared        Whether a field with no explicit request claims
  !>                       its data from shared space.
  function kokkos_shared_default_for_setting( setting ) result(shared)

    implicit none

    character(*), intent(in) :: setting
    logical(l_def) :: shared

#ifdef USE_KOKKOS
    shared = .true._l_def
#else
    shared = .false._l_def
#endif

    if ( trim(adjustl(setting)) == '0' ) shared = .false._l_def

  end function kokkos_shared_default_for_setting

  !> @brief Reports whether a field with no explicit request should claim its
  !>        data from Kokkos shared space.
  !> @details field_mod.t90 is templated to lowercase .f90 and so sees no
  !>          preprocessor macros (see this module's own header); it asks this
  !>          instead of testing USE_KOKKOS. The environment is read once and
  !>          the answer kept, because this is called for every field created
  !>          and because a default that changed part way through a run would
  !>          be far harder to reason about than one that did not.
  !> @return shared   Whether fields are claimed from shared space.
  function kokkos_shared_fields_by_default() result(shared)

    implicit none

    logical(l_def) :: shared

    character(str_def) :: setting
    integer            :: status

    if ( .not. default_decided ) then
      setting = ''
      call get_environment_variable( shared_fields_variable, setting,        &
                                     status=status )
      ! Not set, or too long to be the one value that means anything here.
      if ( status /= 0 ) setting = ''
      default_shared  = kokkos_shared_default_for_setting( setting )
      default_decided = .true._l_def
    end if

    shared = default_shared

  end function kokkos_shared_fields_by_default

  !> @brief Reports whether a released block should be kept rather than given
  !>        back to the allocator.
  !> @details A diagnostic, off unless LFRIC_KOKKOS_SHARED_KEEP is set to 1.
  !>          A premature release and a write past the end of a block both end
  !>          as heap corruption at some later free, and nothing in the crash
  !>          says which happened. Keeping every released block makes the first
  !>          harmless: a run that completes with this set, and refuses some
  !>          releases as it goes, was releasing a block another object still
  !>          held. The environment is read once, for the reason the field
  !>          default gives.
  !> @return keep   Whether a released block is kept alive.
  function keep_released_blocks() result(keep)

    implicit none

    logical(l_def) :: keep

    character(str_def) :: setting
    integer            :: status

    if ( .not. keep_decided ) then
      setting = ''
      call get_environment_variable( keep_released_variable, setting,         &
                                     status=status )
      if ( status /= 0 ) setting = ''
      keep_released = ( trim(adjustl(setting)) == '1' )
      keep_decided  = .true._l_def
    end if

    keep = keep_released

  end function keep_released_blocks

  !> @brief Reports the shared-space bytes currently allocated.
  !> @return bytes   Live shared-space bytes; zero without USE_KOKKOS.
  function kokkos_shared_bytes() result(bytes)

    implicit none

    integer(i_long) :: bytes

#ifdef USE_KOKKOS
    bytes = int( lfric_kokkos_shared_bytes(), i_long )
#else
    bytes = 0_i_long
#endif

  end function kokkos_shared_bytes

  !> @brief Reports the high-water mark of shared-space allocation.
  !> @return bytes   Peak shared-space bytes; zero without USE_KOKKOS.
  function kokkos_shared_peak_bytes() result(bytes)

    implicit none

    integer(i_long) :: bytes

#ifdef USE_KOKKOS
    bytes = int( lfric_kokkos_shared_peak_bytes(), i_long )
#else
    bytes = 0_i_long
#endif

  end function kokkos_shared_peak_bytes

  !> @brief Writes a shared-space summary to standard error.
  !> @details Called on the way out, from driver_kokkos_mod's finalise. It
  !>          writes to standard error rather than through log_mod because
  !>          gungho_model finalises the logger before it finalises Kokkos, so
  !>          there is no logger left to write to.
  !>
  !>          Two lines, answering two different questions. The registry line
  !>          is written by both builds and says how many blocks were claimed
  !>          at once, which is what shows a caller's opt-in set is bounded;
  !>          the allocator line comes from the C++ side and needs USE_KOKKOS,
  !>          and says how many bytes shared space actually issued. A build
  !>          that defines USE_KOKKOS but never reaches a running runtime
  !>          prints a registry line with blocks in it and an allocator line
  !>          with a zero peak, which is the distinction the pair exists to
  !>          make visible.
  subroutine kokkos_shared_report()

    implicit none

    write( error_unit, '(A,I0,A,I0,A)' )                                     &
        'lfric_kokkos_registry: ', n_claimed, ' blocks claimed, ',           &
        peak_claimed, ' at peak'
    flush( error_unit )

#ifdef USE_KOKKOS
    call lfric_kokkos_shared_report()
#endif

  end subroutine kokkos_shared_report

  !> @brief Writes the lifetime figures to standard error.
  !> @details Called from driver_kokkos_mod's finalise, before
  !>          kokkos_shared_release_all, because afterwards the live count is
  !>          zero by construction and says nothing. The two figures answer
  !>          two questions: how many blocks nothing gave back, which should
  !>          be zero once the model has cleared its field collections, and
  !>          how many releases named a block the registry had not issued,
  !>          which should be zero unless a copy escaped the copy rule.
  !>
  !>          A line of its own with a prefix of its own, because the two
  !>          lines kokkos_shared_report writes are matched by
  !>          psy-ir-aidev's tests/test_field_data_is_shared.sh and adding to
  !>          either of them would break that gate. It writes to standard
  !>          error for the same reason kokkos_shared_report does: the logger
  !>          is finalised before Kokkos is.
  subroutine kokkos_shared_report_lifetime()

    implicit none

    write( error_unit, '(A,I0,A,I0,A)' )                                     &
        'lfric_kokkos_lifetime: ', n_claimed, ' blocks live at finalise, ',  &
        n_unknown_releases, ' unknown releases'

    ! A second line, and only when there were any, because it is not a figure
    ! the gates assert: a finaliser that ran on memory no field ever claimed
    ! gave the registry nothing to do, and saying so on the line the gates
    ! read would make a run that met one look like a run that lost a block.
    if ( n_stale_finalisations > 0_i_def ) then
      write( error_unit, '(A,I0,A)' )                                        &
          'lfric_kokkos_lifetime: ', n_stale_finalisations,                  &
          ' finalisations of memory no field had claimed'
    end if
    flush( error_unit )

  end subroutine kokkos_shared_report_lifetime

  !> @brief Reports whether the shared allocator issued a 32-bit real array's
  !>        storage.
  !> @details The question a caller asks before handing an address to code
  !>          that takes a device view of it: a block from ALLOCATE is host
  !>          memory and a block from the shared allocator is reachable from
  !>          both sides, and nothing about the array itself says which.
  !>          kokkos_shared_index cannot answer it, because the zero it
  !>          returns for an address it never issued is the same zero it
  !>          returns for one it issued and no registry entry owns -- the
  !>          state of every block kokkos_shared_allocate hands out.
  !>
  !>          False in a build without USE_KOKKOS, where there is no shared
  !>          allocator, and false for an unassociated or zero-sized array,
  !>          which has no first element to take the address of.
  !> @param [in] array   Array to ask about.
  !> @return owns        Whether the shared allocator issued its storage.
  function kokkos_shared_owns_real32( array ) result(owns)

    implicit none

    real(real32), pointer, intent(in) :: array( : )
    logical(l_def) :: owns

    owns = .false._l_def
    if ( .not. associated(array) ) return
    if ( size(array) < 1 ) return

#ifdef USE_KOKKOS
    owns = ( lfric_kokkos_shared_owns( c_loc(array(1)) ) /= 0_c_int )
#endif

  end function kokkos_shared_owns_real32

  !> @brief Reports whether the shared allocator issued a 64-bit real array's
  !>        storage.
  !> @details As kokkos_shared_owns_real32, for the other real kind.
  !> @param [in] array   Array to ask about.
  !> @return owns        Whether the shared allocator issued its storage.
  function kokkos_shared_owns_real64( array ) result(owns)

    implicit none

    real(real64), pointer, intent(in) :: array( : )
    logical(l_def) :: owns

    owns = .false._l_def
    if ( .not. associated(array) ) return
    if ( size(array) < 1 ) return

#ifdef USE_KOKKOS
    owns = ( lfric_kokkos_shared_owns( c_loc(array(1)) ) /= 0_c_int )
#endif

  end function kokkos_shared_owns_real64

  !> @brief Reports whether the shared allocator issued a 32-bit integer
  !>        array's storage.
  !> @details As kokkos_shared_owns_real32, for the integer kind.
  !> @param [in] array   Array to ask about.
  !> @return owns        Whether the shared allocator issued its storage.
  function kokkos_shared_owns_int32( array ) result(owns)

    implicit none

    integer(int32), pointer, intent(in) :: array( : )
    logical(l_def) :: owns

    owns = .false._l_def
    if ( .not. associated(array) ) return
    if ( size(array) < 1 ) return

#ifdef USE_KOKKOS
    owns = ( lfric_kokkos_shared_owns( c_loc(array(1)) ) /= 0_c_int )
#endif

  end function kokkos_shared_owns_int32

  !> @brief Reports whether the shared allocator has ever issued a block.
  !> @details "Is Kokkos in use" has no single answer, because a build may
  !>          define USE_KOKKOS and still fall back to ALLOCATE for every
  !>          array if the runtime is not running. This answers the decidable
  !>          question instead: it is false in a build without USE_KOKKOS, and
  !>          false in a Kokkos build until the first shared allocation
  !>          succeeds.
  !> @return in_use   Whether any shared-space block has been issued.
  function kokkos_shared_in_use() result(in_use)

    implicit none

    logical(l_def) :: in_use

    in_use = ( kokkos_shared_peak_bytes() > 0_i_long )

  end function kokkos_shared_in_use

end module kokkos_memory_mod
