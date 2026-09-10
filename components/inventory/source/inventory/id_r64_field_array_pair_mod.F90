!-----------------------------------------------------------------------------
! (c) Crown copyright 2022 Met Office. All rights reserved.
! The file LICENCE, distributed with this code, contains details of the terms
! under which the code may be used.
!-----------------------------------------------------------------------------

!> @brief Defines an object to pair field arrays with a unique identifier.
module id_r64_field_array_pair_mod

  use constants_mod,         only: i_def
  use field_real64_mod,      only: field_real64_type
  use function_space_mod,    only: function_space_type
  use id_abstract_pair_mod,  only: id_abstract_pair_type
  use linked_list_data_mod,  only: linked_list_data_type

  implicit none

  private

  ! ========================================================================== !
  ! ID-Field Array Pair
  ! ========================================================================== !

  !> @brief An object pairing a field array with a unique identifier
  !>
  type, public, extends(id_abstract_pair_type) :: id_r64_field_array_pair_type

    private

    type(field_real64_type), allocatable :: field_array_(:)

  contains

    procedure, public :: initialise
    procedure, public :: copy_initialise
    !> Routine a container calls to have the pair copy itself
    procedure, public :: clone
    procedure, public :: get_field_array

    final :: destructor

  end type id_r64_field_array_pair_type

contains

  !> @brief Initialises the id_r64_field_pair object with a new field
  !> @param[in] fs           The function space of the new field
  !> @param[in] id           The integer ID to pair with the field
  !> @param[in] array_size   The size of the field array
  !> @param[in] halo_depth   Optional halo depth for field (to overwrite the
  !!                         default halo depth)
  subroutine initialise(self, fs, id, array_size, halo_depth)

    implicit none

    class(id_r64_field_array_pair_type),      intent(inout) :: self
    type(function_space_type),       pointer, intent(in)    :: fs
    integer(kind=i_def),                      intent(in)    :: id
    integer(kind=i_def),                      intent(in)    :: array_size
    integer(kind=i_def),            optional, intent(in)    :: halo_depth

    integer(kind=i_def) :: i

    allocate(self%field_array_(array_size))

    do i = 1, array_size
      call self%field_array_(i)%initialise(fs, halo_depth = halo_depth)
    end do

    call self%set_id(id)

  end subroutine initialise

  !> @brief Initialises the id_r64_field_array_pair object by copying in fields
  !> @param[in] field_array   The fields to be stored in the paired object
  !> @param[in] id            The integer ID to pair with the field_array
  subroutine copy_initialise(self, field_array, id)

    implicit none

    class(id_r64_field_array_pair_type), intent(inout) :: self
    type(field_real64_type),             intent(in)    :: field_array(:)
    integer(kind=i_def),                 intent(in)    :: id

    integer(kind=i_def) :: i

    allocate(self%field_array_(size(field_array)))

    do i = 1, size(field_array)
      call self%field_array_(i)%initialise(field_array(i)%get_function_space())
      call field_array(i)%copy_field_serial(self%field_array_(i))
    end do

    call self%set_id(id)

  end subroutine copy_initialise

  !> @brief Makes a copy of this pair for a container to keep.
  !>
  !> @details A pair is stored in an inventory's linked list, which used to
  !>          copy its payload with ALLOCATE(..., SOURCE=). That copies each
  !>          held field's data pointer rather than its data, so the stored
  !>          pair and the caller's original named one block of shared space
  !>          per field and the original released them. copy_initialise is
  !>          the deep copy this type already has.
  !>
  !>          A pair whose array was never allocated has nothing to copy;
  !>          copy_initialise cannot be asked for that, so such a pair is
  !>          copied as it stands, which owns nothing.
  !>
  !> @param[in] self   The pair to copy.
  !> @param[out] copy  A newly allocated pair holding copies of the fields.
  subroutine clone(self, copy)

    implicit none

    class(id_r64_field_array_pair_type), intent(in) :: self
    class(linked_list_data_type), pointer, intent(out) :: copy

    type(id_r64_field_array_pair_type), pointer :: new_pair => null()

    allocate( new_pair )

    if ( allocated(self%field_array_) ) then
      call new_pair%copy_initialise( self%field_array_, self%get_id() )
    else
      call new_pair%set_id( self%get_id() )
    end if

    copy => new_pair

  end subroutine clone

  !> @brief Get the field_array corresponding to the paired object
  !> @param[in] self     The paired object
  !> @return             The field_array
  function get_field_array(self) result(field_array)

    implicit none

    class(id_r64_field_array_pair_type), target, intent(in) :: self
    type(field_real64_type),                     pointer    :: field_array(:)

    field_array => self%field_array_

  end function get_field_array

  !> @brief Loops over fields in field array and calls finaliser on each
  subroutine destructor(self)
    implicit none
    type(id_r64_field_array_pair_type), intent(inout) :: self
    integer(kind=i_def) :: i
    if( allocated(self%field_array_) ) then
      do i = 1, size(self%field_array_)
        call self%field_array_(i)%field_final()
      end do
      deallocate(self%field_array_)
    endif
  end subroutine destructor

end module id_r64_field_array_pair_mod
