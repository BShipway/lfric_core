!-----------------------------------------------------------------------------
! (c) Crown copyright 2022 Met Office. All rights reserved.
! The file LICENCE, distributed with this code, contains details of the terms
! under which the code may be used.
!-----------------------------------------------------------------------------

!> @brief Defines an object to pair operators with a unique identifier.
module id_r64_operator_pair_mod

  use constants_mod,         only: i_def
  use function_space_mod,    only: function_space_type
  use operator_real64_mod,   only: operator_real64_type
  use id_abstract_pair_mod,  only: id_abstract_pair_type
  use linked_list_data_mod,  only: linked_list_data_type

  implicit none

  private

  ! ========================================================================== !
  ! ID-Operator Pair
  ! ========================================================================== !

  !> @brief An object pairing a field with a unique identifier
  !>
  type, public, extends(id_abstract_pair_type) :: id_r64_operator_pair_type

    private

    type(operator_real64_type) :: operator_

  contains

    procedure, public :: initialise
    procedure, public :: copy_initialise
    procedure, public :: clone
    procedure, public :: get_operator

    final :: destructor

  end type id_r64_operator_pair_type

contains

  !> @brief Initialises the id_r64_operator_pair object with a new operator
  !> @param[in] fs_target The function space of the target field of the operator
  !> @param[in] fs_source The function space of the source field of the operator
  !> @param[in] id        The integer ID to pair with the operator
  subroutine initialise(self, fs_target, fs_source, id)

    implicit none

    class(id_r64_operator_pair_type),   intent(inout) :: self
    type(function_space_type), pointer, intent(in)    :: fs_target
    type(function_space_type), pointer, intent(in)    :: fs_source
    integer(kind=i_def),                intent(in)    :: id

    call self%operator_%initialise(fs_target, fs_source)
    call self%set_id(id)

  end subroutine initialise

  !> @brief Initialises the id_r64_operator_pair object by copying in an operator
  !> @param[in] operator_in  The operator that will be stored in the paired object
  !> @param[in] id           The integer ID to pair with the operator
  subroutine copy_initialise(self, operator_in, id)

    implicit none

    class(id_r64_operator_pair_type), intent(inout) :: self
    type(operator_real64_type),       intent(in)    :: operator_in
    integer(kind=i_def),              intent(in)    :: id

    ! Copied rather than assigned from deep_copy. An operator whose
    ! stencil comes from Kokkos shared space holds it by pointer, and an
    ! intrinsic assignment copies a pointer component by copying the
    ! pointer; the function result is then finalised, giving back the
    ! very block this pair would be left naming.
    call operator_in%copy_operator_serial(self%operator_)
    call self%set_id(id)

  end subroutine copy_initialise

  !> @brief Makes a copy of this pair for a container to keep.
  !>
  !> @details A pair is stored in an inventory's linked list, which used to
  !>          copy its payload with ALLOCATE(..., SOURCE=). That copies the
  !>          held operator's stencil pointer rather than its values, so the
  !>          stored pair and the caller's local named one block of shared
  !>          space between them and the local released it. copy_initialise is
  !>          the deep copy this type already has.
  !>
  !> @param[in] self   The pair to copy.
  !> @param[out] copy  A newly allocated pair holding a copy of the operator.
  subroutine clone(self, copy)

    implicit none

    class(id_r64_operator_pair_type), intent(in)        :: self
    class(linked_list_data_type), pointer, intent(out) :: copy

    type(id_r64_operator_pair_type), pointer :: new_pair => null()

    allocate( new_pair )

    ! An operator with no stencil yet has nothing to copy, so the copy is left
    ! as it was allocated -- uninitialised, which is the state the one being
    ! copied is in.
    if ( self%operator_%is_initialised() ) then
      call new_pair%copy_initialise( self%operator_, self%get_id() )
    else
      call new_pair%set_id( self%get_id() )
    end if

    copy => new_pair

  end subroutine clone

  !> @brief Get the operator corresponding to the paired object
  !> @param[in] self     The paired object
  !> @return             The operator
  function get_operator(self) result(operator_out)

    implicit none

    class(id_r64_operator_pair_type), target, intent(in) :: self
    type(operator_real64_type),               pointer    :: operator_out

    operator_out => self%operator_

  end function get_operator

  !> @brief Calls finaliser on operator owned by the instance
  subroutine destructor(self)
    implicit none
    type(id_r64_operator_pair_type), intent(inout) :: self
    call self%operator_%operator_final()
  end subroutine destructor

end module id_r64_operator_pair_mod
