!-----------------------------------------------------------------------------
! Copyright (c) 2017,  Met Office, on behalf of HMSO and Queen's Printer
! For further details please refer to the file LICENCE which you
! should have received as part of this distribution.
!-----------------------------------------------------------------------------

!> @brief Linked list data type

!> @details A generic linked list data type - anything that needs to
!>          be put in a linked list must inherit from this

module linked_list_data_mod

  use constants_mod,        only    : i_def

  implicit none

  type, abstract, public :: linked_list_data_type
    private
    integer(i_def) :: id
  contains
    procedure, public :: get_id
    procedure, public :: set_id
    procedure, public :: clone
  end type linked_list_data_type

contains

function get_id(self) result(id)

  implicit none

  class(linked_list_data_type), intent (in)  :: self
  integer(i_def)                             :: id

  id = self%id

end function get_id

!> @brief Makes a copy of this object for a container to keep.
!>
!> @details A linked list stores a copy of what it is given, and used to make
!>          that copy with ALLOCATE(..., SOURCE=). That copies a pointer
!>          component by copying the pointer, so a payload holding storage
!>          through a pointer -- a field whose data was claimed from Kokkos
!>          shared space is the case in hand -- ended up sharing one block
!>          with the caller's original, which then released it. Asking the
!>          payload to copy itself lets a type that owns anything say how.
!>
!>          This default is the SOURCE= that was there before, and is right
!>          for every payload that owns nothing by pointer. A type that holds
!>          a field by value, or a container of fields, overrides it.
!>
!> @param [in] self   The object to copy.
!> @param [out] copy  A newly allocated copy, of the dynamic type of self.
subroutine clone(self, copy)

  implicit none

  class(linked_list_data_type), intent(in)          :: self
  class(linked_list_data_type), pointer, intent(out) :: copy

  allocate( copy, source=self )

end subroutine clone

subroutine set_id(self, id)

  implicit none

  class(linked_list_data_type), intent (inout) :: self
  integer(i_def)  , intent(in)                 :: id

  self%id = id

end subroutine set_id

end module linked_list_data_mod
