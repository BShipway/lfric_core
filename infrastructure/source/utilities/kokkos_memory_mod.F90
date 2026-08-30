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
!>          kokkos_shared_claim allocates a block and keeps owning it, until
!>          kokkos_shared_release_all is called on the way out. That exists
!>          because a Fortran object cannot own a block it holds by pointer:
!>          ALLOCATE with SOURCE= duplicates such an object without duplicating
!>          the storage and without giving the type any way to notice, so the
!>          copy and the original would both free it. Whoever claims a block
!>          therefore states its lifetime by claiming it, and a claimed block
!>          is not reclaimed until the run ends. kokkos_shared_claimed_blocks
!>          reports how many are outstanding, so that a caller opting fields in
!>          can show the count is bounded rather than assert it.
!>
module kokkos_memory_mod

  ! real32, real64 and int32 are the kinds field_mod.t90 is instantiated for.
  ! They come from iso_fortran_env rather than constants_mod because
  ! constants_mod imports them without exporting them.
  use, intrinsic :: iso_fortran_env, only : real32, real64, int32

  use, intrinsic :: iso_c_binding,   only : c_ptr, c_size_t, c_int,          &
                                            c_associated, c_f_pointer,       &
                                            c_loc, c_sizeof

  use constants_mod,                 only : i_def, i_long, l_def

  implicit none

  private

  public :: kokkos_shared_allocate,       &
            kokkos_shared_free,           &
            kokkos_shared_claim,          &
            kokkos_shared_release_all,    &
            kokkos_shared_claimed_blocks, &
            kokkos_shared_bytes,          &
            kokkos_shared_peak_bytes,     &
            kokkos_shared_report,         &
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
  end type shared_block_type

  !> Blocks claimed and not yet released. Grown by doubling.
  type(shared_block_type), allocatable, save :: claimed( : )

  !> How many leading entries of 'claimed' are in use.
  integer(i_def), save :: n_claimed = 0_i_def

  !> The largest n_claimed has ever been. Kept because release_all takes
  !> n_claimed back to zero, so the count a caller needs in order to show its
  !> opt-in set is bounded is gone by the time anything reports it.
  integer(i_def), save :: peak_claimed = 0_i_def

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

  !> @brief Releases an array that kokkos_shared_allocate returned, by
  !>        whichever route it was allocated through.
  interface kokkos_shared_free
    module procedure kokkos_shared_free_real32
    module procedure kokkos_shared_free_real64
    module procedure kokkos_shared_free_int32
  end interface kokkos_shared_free

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
  subroutine remember_block( record )

    implicit none

    type(shared_block_type), intent(in) :: record

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

    if ( n_claimed > peak_claimed ) peak_claimed = n_claimed

  end subroutine remember_block

  !> @brief Claims a 32-bit real array on the caller's behalf.
  !> @details The pointer is disassociated before allocating rather than being
  !>          handed straight to kokkos_shared_allocate, which releases an
  !>          associated pointer on entry. A claimed block belongs to the
  !>          registry, so a caller passing one back in must not have it freed
  !>          from under the registry's record of it.
  !> @param [in,out] array   Pointer to allocate. On return it is associated
  !>                         with 'length' elements, lower bound one.
  !> @param [in] length      Number of elements required.
  subroutine kokkos_shared_claim_real32( array, length )

    implicit none

    real(real32), pointer, intent(inout) :: array( : )
    integer(i_def),        intent(in)    :: length

    type(shared_block_type) :: record

    array => null()
    call kokkos_shared_allocate( array, length )

    record%values_real32 => array
    call remember_block( record )

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
  subroutine kokkos_shared_claim_real64( array, length )

    implicit none

    real(real64), pointer, intent(inout) :: array( : )
    integer(i_def),        intent(in)    :: length

    type(shared_block_type) :: record

    array => null()
    call kokkos_shared_allocate( array, length )

    record%values_real64 => array
    call remember_block( record )

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
  subroutine kokkos_shared_claim_int32( array, length )

    implicit none

    integer(int32), pointer, intent(inout) :: array( : )
    integer(i_def),          intent(in)    :: length

    type(shared_block_type) :: record

    array => null()
    call kokkos_shared_allocate( array, length )

    record%values_int32 => array
    call remember_block( record )

  end subroutine kokkos_shared_claim_int32

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

  !> @brief Writes a one-line shared-space summary to standard error.
  !> @details Called on the way out, from driver_kokkos_mod's finalise. It
  !>          writes to standard error rather than through log_mod because
  !>          gungho_model finalises the logger before it finalises Kokkos, so
  !>          there is no logger left to write to. Does nothing without
  !>          USE_KOKKOS.
  subroutine kokkos_shared_report()

    implicit none

#ifdef USE_KOKKOS
    call lfric_kokkos_shared_report()
#endif

  end subroutine kokkos_shared_report

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
