program iri_driver

use, intrinsic:: iso_fortran_env, only: stderr=>error_unit, stdout=>output_unit

implicit none

logical :: jf(50)
integer, parameter :: jmag = 0
integer :: iyyyy, mmdd, Nalt
real :: glat, glon, dhour
integer :: ymdhms(6)
real:: alt_km_range(3)


real :: oarr(100), outf(20,1000)
real :: tecbo, tecto
real, allocatable :: altkm(:)
character(1024) :: argv
integer :: i
integer :: iParam_Switch

!> jf switch description: https://irimodel.org/IRI-Switches-options.pdf

jf = .true.
!> jf(4:6) = .false. iri2020 default
jf(4:6) = .false.
!> jf(22) = .false. units m^-3
!> jf(23) = .false. iri2020 default
jf(22:23) = .false.
!> jf(30) = .false. iri2020 default
jf(30) = .false.
!> jf(33,35) = .false. iri2020 default
jf(33) = .false.
!> jf(34) = .false. debug messages off
jf(34) = .false.
jf(35) = .false.
!> jf(39,40) = .false. iri2020 default
jf(39:40) = .false.
!> jf(47) = .false. iri2020 default
jf(47) = .false.

! --- command line input
if ((command_argument_count() /= 11) .and. (command_argument_count() /= 16)) then
  write(stderr,*) 'need input parameters: year month day hour minute second glat glon min_alt_km max_alt_km step_alt_km'
  write(stderr,*) 'or need input parameters: year month day hour minute second glat glon min_alt_km max_alt_km step_alt_km 1 f0F2 HmF2 B0 B1'
  write(stderr,*) 'or need input parameters: year month day hour minute second glat glon min_alt_km max_alt_km step_alt_km 2 f107D 107_81 IG Rz'
  stop 1
endif

do i=1,6
  call get_command_argument(i,argv)
  read(argv,*) ymdhms(i)
enddo

call get_command_argument(7, argv)
read(argv,*) glat

call get_command_argument(8, argv)
read(argv,*) glon

do i = 1,3
  call get_command_argument(8+i, argv)
  read(argv,*) alt_km_range(i)
enddo

if (command_argument_count() == 16) then
! Processing optional inputs
  call get_command_argument(12, argv)
  read(argv,*) iParam_Switch
  if (iParam_Switch == 1) then
! User defined foF2, hmF2,B0,B1
    call get_command_argument(13, argv)
    jf(8) = .false.
    read(argv,*) OARR(1)
    call get_command_argument(14, argv)
    jf(9) = .false.
    read(argv,*) OARR(2)
    jf(4) = .false.
    jf(31) = .false.
    call get_command_argument(15, argv)
    jf(43) = .false.
    read(argv,*) OARR(10)
    call get_command_argument(16, argv)
    jf(44) = .false.
    read(argv,*) OARR(35)
  else
! User defined f107D, F107_81, IG12, Rz12
    call get_command_argument(13, argv)
    jf(25) = .false.
    read(argv,*) OARR(41)
    call get_command_argument(14, argv)
    jf(49) = .false.
    read(argv,*) OARR(51)
    call get_command_argument(15, argv)
    jf(27) = .false.
    read(argv,*) OARR(39)
    call get_command_argument(16, argv)
    jf(17) = .false.
    read(argv,*) OARR(33)
  endif
endif

! --- parse
Nalt = int((alt_km_range(2) - alt_km_range(1)) / alt_km_range(3)) + 1
allocate(altkm(Nalt))


altkm(1) = alt_km_range(1)
do i = 2,Nalt
  altkm(i) = altkm(i-1) + alt_km_range(3)
enddo

iyyyy = ymdhms(1)
mmdd = ymdhms(2) * 100 + ymdhms(3)
dhour = ymdhms(4) + ymdhms(5) / 60. + ymdhms(6) / 3600.

!print *, "entering iri_sub"

call read_ig_rz
call readapf107

call IRI_SUB(JF, JMAG, glat, glon, IYYYY, MMDD, DHOUR+25., &
     alt_km_range(1), alt_km_range(2), alt_km_range(3), &
     OUTF,OARR)

!> from irisub.f90:iri_web()
!> IRI2020 call method is totally different from IRI2016.

call IRITEC(glat, glon, jmag, jf, iyyyy, mmdd, dhour+25., &
            0., alt_km_range(2), 0.1, &
            oarr, tecbo, tecto)

! print *, "TRACE: ", tecbo, tecto
oarr(37) = tecbo + tecto
oarr(38) = tecto / oarr(37) * 100

!print '(A,ES10.3,A,F5.1,A)','NmF2 ',oarr(1),' [m^-3]     hmF2 ',oarr(2),' [km] '
!print '(A,F10.3,A,I3,A,F10.3)','F10.7 ',oarr(41), ' Ap ',int(oarr(51)),' B0 ',oarr(10)

!print *,'Altitude    Ne    O2+'
do i = 1,Nalt
  print '(F10.3, 11ES16.8)', altkm(i), outf(:11,i)
enddo


print '(/,100ES16.8)', oarr

end program
