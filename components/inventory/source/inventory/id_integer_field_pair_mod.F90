!-----------------------------------------------------------------------------
! (c) Crown copyright 2022 Met Office. All rights reserved.
! The file LICENCE, distributed with this code, contains details of the terms
! under which the code may be used.
!-----------------------------------------------------------------------------

!> @brief Defines an object to pair integer fields with a unique identifier.
module id_integer_field_pair_mod

  use constants_mod,         only: i_def, l_def, str_def
  use function_space_mod,    only: function_space_type
  use integer_field_mod,     only: integer_field_type
  use id_abstract_pair_mod,  only: id_abstract_pair_type
  use linked_list_data_mod,  only: linked_list_data_type

  implicit none

  private

  ! ========================================================================== !
  ! ID-Field Pair
  ! ========================================================================== !

  !> @brief An object pairing an integer field with a unique identifier
  !>
  type, public, extends(id_abstract_pair_type) :: id_integer_field_pair_type

    private

    type(integer_field_type) :: field_

  contains

    procedure, public :: initialise
    procedure, public :: copy_initialise
    !> Routine a container calls to have the pair copy itself
    procedure, public :: clone
    procedure, public :: get_field

    final :: destructor

  end type id_integer_field_pair_type

contains

  !> @brief Initialises the id_integer_field_pair object with a new field
  !> @param[in] fs           The function space of the new field
  !> @param[in] id           The integer ID to pair with the field
  !> @param[in] name         Optional name to give to field
  !> @param[in] halo_depth   Optional halo depth for field (to overwrite the
  !!                         default halo depth)
  subroutine initialise(self, fs, id, name, halo_depth)

    implicit none

    class(id_integer_field_pair_type),  intent(inout) :: self
    type(function_space_type), pointer, intent(in)    :: fs
    integer(kind=i_def),                intent(in)    :: id
    character(*),             optional, intent(in)    :: name
    integer(kind=i_def),      optional, intent(in)    :: halo_depth

    character(str_def) :: local_name

    if ( present(name) ) then
      local_name = name
    else
      local_name = 'none'
    end if

    call self%field_%initialise(fs, name=local_name, &
                                halo_depth=halo_depth)
    call self%set_id(id)

  end subroutine initialise

  !> @brief Initialises the id_integer_field_pair object by copying in a field
  !> @param[in] field    The field that will be stored in the paired object
  !> @param[in] id       The integer ID to pair with the field
  subroutine copy_initialise(self, field, id)

    implicit none

    class(id_integer_field_pair_type), intent(inout) :: self
    type(integer_field_type),          intent(in)    :: field
    integer(kind=i_def),           intent(in)    :: id

    call self%field_%initialise(field%get_function_space(), &
                                halo_depth=field%get_field_halo_depth())
    call field%copy_field_serial(self%field_)
    call self%set_id(id)

  end subroutine copy_initialise

  !> @brief Makes a copy of this pair for a container to keep.
  !>
  !> @details A pair is stored in an inventory's linked list, which used to
  !>          copy its payload with ALLOCATE(..., SOURCE=). That copies the
  !>          held field's data pointer rather than its data, so the stored
  !>          pair and the caller's original named one block of shared space
  !>          between them and the original released it. copy_initialise is
  !>          the deep copy this type already has.
  !>
  !> @param[in] self   The pair to copy.
  !> @param[out] copy  A newly allocated pair holding a copy of the field.
  subroutine clone(self, copy)

    implicit none

    class(id_integer_field_pair_type), intent(in) :: self
    class(linked_list_data_type), pointer, intent(out) :: copy

    type(id_integer_field_pair_type), pointer :: new_pair => null()

    allocate( new_pair )

    call new_pair%copy_initialise( self%field_, self%get_id() )

    copy => new_pair

  end subroutine clone

  !> @brief Get the field corresponding to the paired object
  !> @param[in] self     The paired object
  !> @return             The field
  function get_field(self) result(field)

    implicit none

    class(id_integer_field_pair_type), target, intent(in) :: self
    type(integer_field_type),                  pointer    :: field

    field => self%field_

  end function get_field

  !> @brief Calls finaliser on the field owned by the instance
  subroutine destructor(self)
    implicit none
    type(id_integer_field_pair_type), intent(inout) :: self
    call self%field_%field_final()
  end subroutine destructor

end module id_integer_field_pair_mod
