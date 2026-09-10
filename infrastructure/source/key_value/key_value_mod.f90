!-----------------------------------------------------------------------------
! (C) Crown copyright 2023 Met Office. All rights reserved.
! The file LICENCE, distributed with this code, contains details of the terms
! under which the code may be used.
!-----------------------------------------------------------------------------
!
!> @brief   Defines key-value pair objects.
!> @details An abstract key-value pair object is defined (key_value_type) along
!>          with the following concrete types.
!>
!> * key_value_type (abstract)
!>     - int32_key_value_type,       int64_key_value_type
!>     - real32_key_value_type,      real64_key_value_type
!>     - logical_key_value_type,     str_key_value_type
!>     - int32_arr_key_value_type,   int64_arr_key_value_type
!>     - real32_arr_key_value_type,  real64_arr_key_value_type
!>     - logical_arr_key_value_type, str_arr_key_value_type
!>     - abstract_key_value_type
!>
!> Concrete types can only be initialised once, however the value is directly
!> accessible without a getter/setter.
!----------------------------------------------------------------------------
module key_value_mod

  use, intrinsic :: iso_fortran_env, only: int32, int64, real32, real64

  use constants_mod,        only: imdi, cmdi, str_longlong, str_def
  use linked_list_data_mod, only: linked_list_data_type
  use log_mod,              only: log_event, log_scratch_space, &
                                  LOG_LEVEL_INFO, LOG_LEVEL_ERROR

  implicit none

  private

  public :: key_value_type,            abstract_value_type
  public :: int32_key_value_type,      int64_key_value_type
  public :: int32_arr_key_value_type,  int64_arr_key_value_type
  public :: real32_key_value_type,     real64_key_value_type
  public :: real32_arr_key_value_type, real64_arr_key_value_type
  public :: logical_key_value_type,    logical_arr_key_value_type
  public :: str_key_value_type,        str_arr_key_value_type
  public :: abstract_key_value_type
  public :: create_key_value, create_key_value_sca, create_key_value_arr

  interface create_key_value
   module procedure create_key_value_sca, create_key_value_arr
  end interface

  !=========================================
  ! Key-Value pair abstract type
  !=========================================
  type, abstract, extends(linked_list_data_type) :: key_value_type

    private
    character(:), allocatable :: key

  contains
    procedure :: key_value_initialise
    procedure :: get_key => get_key_value_key
  end type key_value_type

  !=======================================================================
  ! Abstract type that is the parent to any object held in key-value pair
  !=======================================================================
  type, abstract :: abstract_value_type
  contains
  end type abstract_value_type


  !===============================================
  ! Concrete Types of Abstract: key_value_type
  !===============================================

  !===============================================
  ! int32_key_value_type: 32-bit integer key-value pair
  !===============================================
  type, extends(key_value_type) :: int32_key_value_type
    integer(int32) :: value = -huge(0_int32)
  contains
    procedure :: initialise => init_int32_key_value
  end type int32_key_value_type

  !===============================================
  ! int64_key_value_type: 64-bit integer key-value pair
  !===============================================
  type, extends(key_value_type) :: int64_key_value_type
    integer(int64) :: value = -huge(0_int64)
  contains
    procedure :: initialise => init_int64_key_value
  end type int64_key_value_type

  !===============================================
  ! real32_key_value_type: 32-bit real key-value pair
  !===============================================
  type, extends(key_value_type) :: real32_key_value_type
    real(real32) :: value = -huge(0.0_real32)
  contains
    procedure :: initialise => init_real32_key_value
  end type real32_key_value_type

  !==============================================
  ! real64_key_value_type: 64-bit real key-value pair
  !==============================================
  type, extends(key_value_type) :: real64_key_value_type
    real(real64) :: value = -huge(0.0_real64)
  contains
    procedure :: initialise => init_real64_key_value
  end type real64_key_value_type

  !==============================================
  ! logical_key_value_type: Logical key-value pair
  !==============================================
  type, extends(key_value_type) :: logical_key_value_type
    logical :: value = .false.
  contains
    procedure :: initialise => init_logical_key_value
  end type logical_key_value_type

  !==============================================
  ! str_key_value_type: string key-value pair
  !==============================================
  type, extends(key_value_type) :: str_key_value_type
    character(str_longlong) :: value
  contains
    procedure :: initialise => init_str_key_value
  end type str_key_value_type

  !===============================================
  ! int32_key_value_type: 32-bit integer key-value pair
  !===============================================
  type, extends(key_value_type) :: int32_arr_key_value_type
    integer(int32), allocatable :: value(:)
  contains
    procedure :: initialise => init_int32_arr_key_value
  end type int32_arr_key_value_type

  !===============================================
  ! int64_key_value_type: 64-bit integer key-value pair
  !===============================================
  type, extends(key_value_type) :: int64_arr_key_value_type
    integer(int64), allocatable :: value(:)
  contains
    procedure :: initialise => init_int64_arr_key_value
  end type int64_arr_key_value_type

  !===============================================
  ! int32_key_value_type: 32-bit integer key-value pair
  !===============================================
  type, extends(key_value_type) :: real32_arr_key_value_type
    real(real32), allocatable :: value(:)
  contains
    procedure :: initialise => init_real32_arr_key_value
  end type real32_arr_key_value_type

  !===============================================
  ! int64_key_value_type: 64-bit integer key-value pair
  !===============================================
  type, extends(key_value_type) :: real64_arr_key_value_type
    real(real64), allocatable :: value(:)
  contains
    procedure :: initialise => init_real64_arr_key_value
  end type real64_arr_key_value_type

  !==============================================
  ! logical_arr_key_value_type: Logical key-value pair
  !==============================================
  type, extends(key_value_type) :: logical_arr_key_value_type
    logical, allocatable :: value(:)
  contains
    procedure :: initialise => init_logical_arr_key_value
  end type logical_arr_key_value_type

  !==============================================
  ! str_arr_key_value_type: string array key-value pair
  !==============================================
  type, extends(key_value_type) :: str_arr_key_value_type
    !> @todo This was applied with #3547. This would have been
    !>       similar to the scalar string: i.e.
    !>
    !>       character(str_longlong), allocatable :: value(:)
    !>
    !>       However, the revision of the Intel compiler on the XC40
    !>       produced unexpected behaviour so the length has been
    !>       limited to str_def. This should be revisited when the
    !>       XC40 compilers are later than 17.0.0.098/5.
    character(str_def), allocatable :: value(:)
  contains
    procedure :: initialise => init_str_arr_key_value
  end type str_arr_key_value_type

  !===============================================
  ! abstract_key_value_type: abstract object key_value pair
  !===============================================
  type, extends(key_value_type) :: abstract_key_value_type
    class(abstract_value_type), allocatable :: value
  contains
    procedure :: initialise => init_abstract_key_value
    procedure :: set_value  => set_abstract_key_value
    procedure :: clone      => clone_abstract_key_value
  end type abstract_key_value_type

contains


!> Instantiates correct object for given scalar value type.
!>
!> Note that this routine allocates a pointer but it does not manage the
!> freeing of that pointer. That is the calling routine's responsibility if
!> memory leaks are to be avoided.
!>
!> @todo Fortran "block" syntax was used to keep scoping neat but in order to
!>       make progress in face of an fParser which doesn't recognise the syntax
!>       they were removed. When fParser is fixed they could be put back again.
!>
!> @param [in] key      String used for key
!> @param [in] value    Scalar value to be stored in key-value pair
!>
function create_key_value_sca( key, value ) result(instance)

  implicit none

  character(*), intent(in) :: key
  class(*),     intent(in) :: value

  class(key_value_type), pointer :: instance

  type(int32_key_value_type)    :: concrete_int32
  type(int64_key_value_type)    :: concrete_int64
  type(real32_key_value_type)   :: concrete_real32
  type(real64_key_value_type)   :: concrete_real64
  type(logical_key_value_type)  :: concrete_logical
  type(str_key_value_type)      :: concrete_str

  select type (value)

  type is (integer(int32))
    call concrete_int32%initialise( key, value )
    allocate( instance, source=concrete_int32 )

  type is (integer(int64))
    call concrete_int64%initialise( key, value )
    allocate( instance, source=concrete_int64 )

  type is (real(real32))
    call concrete_real32%initialise( key, value )
    allocate( instance, source=concrete_real32 )

  type is (real(real64))
    call concrete_real64%initialise( key, value )
    allocate( instance, source=concrete_real64 )

  type is (logical)
    call concrete_logical%initialise( key, value )
    allocate( instance, source=concrete_logical )

  type is (character(*))
    call concrete_str%initialise( key, value )
    allocate( instance, source=concrete_str )

  class is (abstract_value_type)
    ! Built where it will live rather than in a local that is then copied by
    ! SOURCE=. The value may hold an LFRic field, whose data is a block the
    ! field gives back when it is finalised; a copy taken into a local
    ! duplicates the pointer and not the block, and the local's destruction
    ! at the end of this function would give back a block the copy is still
    ! using. See set_abstract_key_value.
    allocate( abstract_key_value_type :: instance )
    select type ( instance )
    type is ( abstract_key_value_type )
      call instance%initialise( key, value )
    end select

  class default
    write( log_scratch_space, &
           '("Unable to pair key ''", A, "'' with unsupported value")' ) key
    call log_event( log_scratch_space, log_level_error )
  end select

end function create_key_value_sca


!> Instantiates correct object for given array value type.
!>
!> Note that this routine allocates a pointer but it does not manage the
!> freeing of that pointer. That is the calling routine's responsibility if
!> memory leaks are to be avoided.
!>
!> @param [in] key      String used for key
!> @param [in] value    Array value to be stored in key-value pair
function create_key_value_arr( key, value ) result(instance)

  implicit none

  character(*), intent(in) :: key
  class(*),     intent(in) :: value(:)

  class(key_value_type), pointer :: instance

  type(int32_arr_key_value_type)    :: concrete_int32_arr
  type(int64_arr_key_value_type)    :: concrete_int64_arr
  type(real32_arr_key_value_type)   :: concrete_real32_arr
  type(real64_arr_key_value_type)   :: concrete_real64_arr
  type(logical_arr_key_value_type)  :: concrete_logical_arr
  type(str_arr_key_value_type)      :: concrete_str_arr

  select type (value)

  type is (integer(int32))
    call concrete_int32_arr%initialise( key, value )
    allocate( instance, source=concrete_int32_arr )

  type is (integer(int64))
    call concrete_int64_arr%initialise( key, value )
    allocate( instance, source=concrete_int64_arr )

  type is (real(real32))
    call concrete_real32_arr%initialise( key, value )
    allocate( instance, source=concrete_real32_arr )

  type is (real(real64))
    call concrete_real64_arr%initialise( key, value )
    allocate( instance, source=concrete_real64_arr )

  type is (logical)
    call concrete_logical_arr%initialise( key, value )
    allocate( instance, source=concrete_logical_arr )

  type is (character(*))
    call concrete_str_arr%initialise( key, value )
    allocate( instance, source=concrete_str_arr )

  class default
    write( log_scratch_space, &
           '("Unable to pair key ''", A, "'' with unsupported value")' ) key
    call log_event( log_scratch_space, log_level_error )
  end select

end function create_key_value_arr

!=========================================
! Abstract Initialiser
!=========================================
subroutine key_value_initialise( self, key )

  implicit none

  class(key_value_type), intent(inout) :: self
  character(*),          intent(in)    :: key

  if ( allocated(self%key) ) then
    write( log_scratch_space,'(A)' ) &
        'Type already initialised as ' // trim(self%key)
    call log_event( log_scratch_space, LOG_LEVEL_ERROR )
  else
    allocate( self%key, source=trim(key) )
  end if

  return
end subroutine key_value_initialise


!> @brief Returns key string for the pair
function get_key_value_key( self ) result( key )

  implicit none

  class(key_value_type), intent(in) :: self

  character(:), allocatable :: key

  allocate( key, source=trim(self%key) )

  return
end function get_key_value_key


!=================================================
! Concrete initialisers
! These set the key/value of the concrete types
!=================================================

!> @brief Initialiser for 32-bit integer key-value pair object
!> @param [in] key      String used for key
!> @param [in] value    32-bit integer value
subroutine init_int32_key_value( self, key, value )

  implicit none

  class(int32_key_value_type), intent(inout) :: self

  character(*),   intent(in) :: key
  integer(int32), intent(in) :: value

  call self%key_value_initialise( key )
  self%value = value

  return
end subroutine init_int32_key_value

!> @brief Initialiser for 64-bit integer key-value pair object
!> @param [in] key      String used for key
!> @param [in] value    64-bit integer value
subroutine init_int64_key_value( self, key, value )

  implicit none

  class(int64_key_value_type), intent(inout) :: self

  character(*),   intent(in) :: key
  integer(int64), intent(in) :: value

  call self%key_value_initialise( key )
  self%value = value

  return
end subroutine init_int64_key_value

!> @brief Initialiser for 32-bit real key-value pair object
!> @param [in] key    String used for key
!> @param [in] value  32-bit real value
subroutine init_real32_key_value( self, key, value )

  implicit none

  class(real32_key_value_type), intent(inout) :: self

  character(*), intent(in) :: key
  real(real32), intent(in) :: value

  call self%key_value_initialise( key )
  self%value = value

  return
end subroutine init_real32_key_value

!> @brief Initialiser for 64-bit real key-value pair object
!> @param [in] key    String used for key
!> @param [in] value  64-bit real value
subroutine init_real64_key_value( self, key, value )

  implicit none

  class(real64_key_value_type), intent(inout) :: self

  character(*), intent(in) :: key
  real(real64), intent(in) :: value

  call self%key_value_initialise( key )
  self%value = value

  return
end subroutine init_real64_key_value

!> @brief Initialiser for logical key-value pair object
!> @param [in] key    String used for key
!> @param [in] value  Logical value
subroutine init_logical_key_value( self, key, value )

  implicit none

  class(logical_key_value_type), intent(inout) :: self

  character(*), intent(in) :: key
  logical,      intent(in) :: value

  call self%key_value_initialise( key )
  self%value = value

  return
end subroutine init_logical_key_value

!> @brief Initialiser for string key-value pair object
!> @param [in] key    String used for key
!> @param [in] value  String value
subroutine init_str_key_value( self, key, value )

  implicit none

  class(str_key_value_type), intent(inout) :: self

  character(*), intent(in) :: key
  character(*), intent(in) :: value

  call self%key_value_initialise( key )
  self%value = trim(value)

  return
end subroutine init_str_key_value

!> @brief Initialiser for 32-bit integer array key-value pair object
!> @param [in] key      String used for key
!> @param [in] value    32-bit integer array values
subroutine init_int32_arr_key_value( self, key, value )

  implicit none

  class(int32_arr_key_value_type), intent(inout) :: self

  character(*),   intent(in) :: key
  integer(int32), intent(in) :: value(:)

  call self%key_value_initialise( key )
  allocate( self%value, source=value )

  return
end subroutine init_int32_arr_key_value

!> @brief Initialiser for 64-bit integer array key-value pair object
!> @param [in] key      String used for key
!> @param [in] value    64-bit integer array values
subroutine init_int64_arr_key_value( self, key, value )

  implicit none

  class(int64_arr_key_value_type), intent(inout) :: self

  character(*),   intent(in) :: key
  integer(int64), intent(in) :: value(:)

  call self%key_value_initialise( key )
  allocate( self%value, source=value )

  return
end subroutine init_int64_arr_key_value

!> @brief Initialiser for 32-bit real array key-value pair object
!> @param [in] key      String used for key
!> @param [in] value    32-bit real array values
subroutine init_real32_arr_key_value( self, key, value )

  implicit none

  class(real32_arr_key_value_type), intent(inout) :: self

  character(*),   intent(in) :: key
  real(real32), intent(in) :: value(:)

  call self%key_value_initialise( key )
  allocate( self%value, source=value )

  return
end subroutine init_real32_arr_key_value

!> @brief Initialiser for 64-bit real array key-value pair object
!> @param [in] key      String used for key
!> @param [in] value    64-bit real array values
subroutine init_real64_arr_key_value( self, key, value )

  implicit none

  class(real64_arr_key_value_type), intent(inout) :: self

  character(*),   intent(in) :: key
  real(real64), intent(in) :: value(:)

  call self%key_value_initialise( key )
  allocate( self%value, source=value )

  return
end subroutine init_real64_arr_key_value

!> @brief Initialiser for a logical array key-value pair object
!> @param [in] key      String used for key
!> @param [in] value    logical array values
subroutine init_logical_arr_key_value( self, key, value )

  implicit none

  class(logical_arr_key_value_type), intent(inout) :: self

  character(*), intent(in) :: key
  logical,      intent(in) :: value(:)

  call self%key_value_initialise( key )
  allocate( self%value(size(value)) )
  self%value(:) = value(:)

  return
end subroutine init_logical_arr_key_value

!> @brief Initialiser for string key-value pair object
!> @param [in] key      String used for key
!> @param [in] value    String array values
subroutine init_str_arr_key_value( self, key, value )

  implicit none

  class(str_arr_key_value_type), intent(inout) :: self

  character(*), intent(in) :: key
  character(*), intent(in) :: value(:)

  integer :: arr_len, i

  arr_len  = size(value)

  call self%key_value_initialise( key )
  allocate( self%value(arr_len) )
  do i=1, arr_len
    self%value(i) = trim(value(i))
  end do

  return
end subroutine init_str_arr_key_value

!> @brief Initialiser for abstract-object key-value pair object
!> @param [in] key      String used for key
!> @param [in] value    Abstract object value
subroutine init_abstract_key_value( self, key, value )

  implicit none

  class(abstract_key_value_type), intent(inout) :: self

  character(*), intent(in) :: key
  class(abstract_value_type), intent(in) :: value

  call self%key_value_initialise( key )

  call self%set_value( value )

  return
end subroutine init_abstract_key_value

!> @brief Gives the pair the value it is to hold.
!>
!> @details This is the one copy of the value that is ever made, and it is
!>          made straight into the pair that will keep it. That matters
!>          because a type derived from abstract_value_type may hold an LFRic
!>          field by value, and a field holds its data through a pointer to a
!>          block it owns and gives back when it is finalised. ALLOCATE with
!>          SOURCE= copies the pointer and not the block, so every extra copy
!>          of such a value is another object that would give the same block
!>          back. Splitting the value out of initialise lets
!>          key_value_collection_mod put an empty pair into its list -- which
!>          copies the pair -- and fill it afterwards, so that no copy of the
!>          value is ever made into an object that is about to be destroyed.
!>
!> @param [in] value  The value to copy into the pair.
subroutine set_abstract_key_value( self, value )

  implicit none

  class(abstract_key_value_type), intent(inout) :: self

  class(abstract_value_type), intent(in) :: value

  if ( allocated(self%value) ) then
    call log_event( 'Key-value pair already holds a value', LOG_LEVEL_ERROR )
  end if

  allocate(self%value, source=value)

  return
end subroutine set_abstract_key_value

!> @brief Makes the copy a container keeps of this pair.
!>
!> @details A linked list stores a copy of what it is given and asks the
!>          payload to make it (linked_list_data_mod's clone). The default
!>          copies by SOURCE=, which is right for a pair holding nothing, and
!>          wrong for one already holding a value that owns storage through a
!>          pointer: the copy and the original would name one block and each
!>          would give it back. No deep copy of an arbitrary
!>          abstract_value_type exists -- the type has no copy protocol -- so
!>          rather than make a copy that cannot be right, a pair is expected
!>          to be inserted empty and filled afterwards through set_value,
!>          which is what key_value_collection_mod does. Inserting a full one
!>          is refused here instead of becoming a double release later.
!>
!> @param [out] copy  A newly allocated copy of this pair.
subroutine clone_abstract_key_value( self, copy )

  implicit none

  class(abstract_key_value_type), intent(in)         :: self
  class(linked_list_data_type), pointer, intent(out) :: copy

  if ( allocated(self%value) ) then
    call log_event( 'A key-value pair holding an object cannot be copied '// &
                    'into a collection: insert it empty and set its value',  &
                    LOG_LEVEL_ERROR )
  end if

  allocate( copy, source=self )

  return
end subroutine clone_abstract_key_value

end module key_value_mod
