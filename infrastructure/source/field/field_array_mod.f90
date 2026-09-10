!-----------------------------------------------------------------------------
! (C) Crown copyright 2023 Met Office. All rights reserved.
! The file LICENCE, distributed with this code, contains details of the terms
! under which the code may be used.
!-----------------------------------------------------------------------------

!>@brief An object for holding a named field bundle.
!>
!> @details This is an object that encapsulates a field bundle with its
!>          name. A field bundle is just a fancy name for an array of fields
module field_array_mod

  use constants_mod,           only: i_def, str_def
  use field_mod,               only: field_type
  use linked_list_data_mod,    only: linked_list_data_type
  use pure_abstract_field_mod, only: pure_abstract_field_type

  implicit none

  private

  type, extends(pure_abstract_field_type), public :: field_array_type
    private
    !> An array of fields
    type(field_type ),allocatable, public :: bundle(:)
    !> Name of the bundle. Note the name is immutable once defined via
    !! the initialiser.
    character(str_def)                    :: name
  contains
    !> Initialiser for a field array. May only be called once.
    procedure, public :: initialise => field_array_initialiser
    !> Returns the name of the field bundle
    procedure, public :: get_name
    !> Routine a container calls to have the field array copy itself
    procedure, public :: clone => field_array_clone
    !> Clear field array object, deallocating the bundle held within
    procedure, public :: clear
    !> Finaliser for a field pointer object
    final :: destructor
  end type field_array_type
contains

  !> Initialise a <code>field_array</code> object.
  !>
  !> @param [in] length The number of elements of bundle to allocate
  !> @param [in] name The field bundle name.
  !>
  subroutine field_array_initialiser(self, &
                                     length,  &
                                     name)
    implicit none

    class(field_array_type), intent(inout) :: self
    integer(i_def),           intent(in)   :: length
    character(*), optional,   intent(in)   :: name

    if ( present(name) ) then
      self%name = name
    else
      self%name = 'none'
    end if

    allocate(self%bundle(length))

  end subroutine field_array_initialiser

  !> Returns the name of the field bundle
  !> @return name The name of the field bundle
  function get_name(self) result(name)
    implicit none

    class(field_array_type), intent(in) :: self
    character(str_def) :: name

    name = self%name

  end function get_name

  !> @brief Make a copy of this field array for a container to keep.
  !>
  !> @details The default clone copies the payload with
  !>          ALLOCATE(..., SOURCE=), which would copy each member field's
  !>          override_data by copying the pointer, leaving the copy and this
  !>          object naming one block per member. Each member is copied the
  !>          only way a field may be, through copy_field_serial, and so gets
  !>          a block of its own.
  !>
  !>          A member that has not been initialised has nothing to copy and
  !>          is left as it was allocated. That is a state the model puts a
  !>          bundle in: a field array is added to a collection with its
  !>          members filled in afterwards.
  !>
  !> @param [in] self   The field array to copy.
  !> @param [out] copy  A newly allocated field array holding the same values.
  subroutine field_array_clone(self, copy)

    implicit none

    class(field_array_type), intent(in) :: self
    class(linked_list_data_type), pointer, intent(out) :: copy

    type(field_array_type), pointer :: new_array => null()
    integer(i_def)                  :: i

    allocate( new_array )

    new_array%name = self%name

    if ( allocated(self%bundle) ) then
      allocate( new_array%bundle(size(self%bundle)) )
      do i = 1, size(self%bundle, kind=i_def)
        if ( self%bundle(i)%is_initialised() ) then
          call self%bundle(i)%copy_field_serial( new_array%bundle(i) )
        end if
      end do
    end if

    call new_array%set_id( self%get_id() )

    copy => new_array

  end subroutine field_array_clone

  !> Clear field array object, deallocating the bundle held within
  subroutine clear(self)
    implicit none

    class(field_array_type), intent(inout) :: self

    integer(kind=i_def) :: i

    if (allocated(self%bundle)) then
      do i = 1, size(self%bundle)
        call self%bundle(i)%field_final()
      end do
      deallocate(self%bundle)
    endif

  end subroutine clear

  !> Finaliser for the field array.
  subroutine destructor(self)
    implicit none

    type(field_array_type), intent(inout) :: self

    call self%clear()

  end subroutine destructor

end module field_array_mod

