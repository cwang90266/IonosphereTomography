program iri2020_namelist_driver

use, intrinsic:: iso_fortran_env, only: stderr=>error_unit, stdout=>output_unit

implicit none

! Default size of arrays for inputs.
integer, parameter :: max_npts=1000
integer, parameter :: max_nheight=1000
integer, parameter :: max_nParam=9
integer, parameter :: max_nSample=5000

! Definition of input variables in the namelist
integer :: npts, nheight, nSample
integer :: idxHour,idxF107D,idxap,idxIG12,idxRz12,idxfoF2,idxHmF2,idxB0,idxB1
real :: height_grid(max_nheight)
real :: phy_inputs(max_nParam,max_nSample)
real :: latitude(max_npts), longitude(max_npts)
real :: fill_value
integer :: year, month, day, hour, minute,second
NAMELIST/EDPSamples/npts, nheight, nSample, height_grid,latitude,longitude, &
        idxHour,idxF107D,idxap,idxIG12,idxRz12,idxfoF2,idxHmF2,idxB0,idxB1, &
        year, month, day, hour, minute,second,phy_inputs,fill_value

! Local working variables
character(1024) :: argv
character(132) :: namelist_filename, output_filename
logical :: jf(50)
real :: oarr(100), outf(20,1000)
integer :: i, mmdd, iPts, iSample
real :: dhour, dhour_default
integer, parameter :: JMAG = 0
integer :: NaN_Flag

! --- command line inputs are the file names for namelist and binary output
if (command_argument_count() /= 2)  then
  write(stderr,*) 'need input parameters: filenames for namelist input and binary output'
  stop 1
endif
! Processing command line arguments
call get_command_argument(1,argv)
read(argv,*) namelist_filename
call get_command_argument(2,argv)
read(argv,*) output_filename

! Open the namelist
OPEN(UNIT=10,FILE=namelist_filename,status='OLD')
READ(10,EDPSamples)
CLOSE(UNIT=10)
!write(*,*) 'Completed reading namelist'
! Set inputs to irisub that are common to all spatial points and samples
mmdd = month * 100 + day
dhour_default = hour + minute / 60. + second / 3600.

!> jf switch description: https://irimodel.org/IRI-Switches-options.pdf
jf = .true.
jf(4:6) = .false.
jf(22:23) = .false.
jf(30) = .false.
jf(33) = .false.
jf(34) = .false.
!
! Turn on estorm
!jf(35) = .false.
!
jf(39:40) = .false.
jf(47) = .false.

call read_ig_rz
call readapf107

! Open binary output file and write the header
OPEN(UNIT=20, FILE=output_filename,STATUS='NEW', ACCESS='STREAM')
WRITE(20) npts, nSample, nheight
! write(*,*) 'Complete open data file and write neaders'
! Loop over all spatial points and samples
do iSample =1, nSample
   do iPts = 1, npts
! Assign parameter sample values
! Setting logical flags for parameter option selection
    if (idxHour /= -1 .and. phy_inputs(idxHour,iSample) > fill_value) then
        dhour =  phy_inputs(idxHour,iSample) + minute / 60. + second / 3600.
    ELSE
        dhour = dhour_default
    endif

    if (idxap /= -1 .and. phy_inputs(idxap,iSample) > fill_value) then
      OARR(51) = phy_inputs(idxap,iSample)
      jf(49) = .false.
    ELSE
      jf(49) = .true.
    endif

    if (idxf107D /= -1 .and. phy_inputs(idxF107D,iSample) > fill_value) then
      OARR(41) = phy_inputs(idxF107D,iSample)
      jf(25) = .false.
    ELSE
      jf(25) = .true.
    endif
    if (idxIG12 /= -1 .and. phy_inputs(idxIG12,iSample) > fill_value) then
      OARR(39) = phy_inputs(idxIG12,iSample)
      jf(27) = .false.
    ELSE
      jf(27) = .true.
    endif
    if (idxRz12 /= -1 .and. phy_inputs(idxRz12,iSample) > fill_value) then
      OARR(33) = phy_inputs(idxRz12,iSample)
      jf(17) = .false.
    ELSE
      jf(17) = .true.
    endif
    if (idxfoF2 /= -1 .and. phy_inputs(idxfoF2,iSample) > fill_value) then
      OARR(1) = phy_inputs(idxfoF2,iSample)
      jf(8) = .false.
    ELSE
      jf(8) = .true.
    endif
    if (idxHmF2 /= -1 .and. phy_inputs(idxHmF2,iSample) > fill_value) then
      OARR(2) = phy_inputs(idxHmF2,iSample)
      jf(9) = .false.
    else 
      jf(9) = .true.
    endif
    if (idxB0 /= -1 .and. phy_inputs(idxB0,iSample)> fill_value) then
      OARR(43) = phy_inputs(idxB0,iSample)
      jf(4) = .false.
      jf(31) = .false.
    else
      jf(4) = .true.
      jf(31) = .true.      
    endif
    if (idxB1 /= -1 .and. phy_inputs(idxB1,iSample) > fill_value) then
      OARR(44) = phy_inputs(idxB1,iSample)
      jf(4) = .false.
      jf(44) = .false.
    elseif (idxB0 /= -1 .and. phy_inputs(idxB0,iSample)> fill_value) then
      jf(44) = .true.
    else
      jf(4) = .true.
      jf(44) = .true.
    endif    
!    write(*,*) 'Inside loop ', iSample, iPts
!    write(*,*) JF, JMAG, latitude(iPts), longitude(iPts)
!    write(*,*) year, mmdd, dhour+25.,nheight, height_grid
    call IRISUB_Height_Grid(JF, JMAG, latitude(iPts), longitude(iPts), &
        year, mmdd, dhour+25., nheight, height_grid, outf,OARR)
! Write electron density for a given sample at a given point
    NaN_Flag = 0 
    DO i=1, nheight 
        if (ISNAN(outf(1,i)))  NaN_Flag =1
    ENDDO
    if (NaN_Flag == 1) THEN
        WRITE(*,*) 'Ne contains NaN for sample, point ', iSample, iPts
    ENDIF   
    WRITE(20) (outf(1,i),i=1,nheight)
    WRITE(20) (OARR(i),i=1,12), OARR(35)
  enddo
enddo

CLOSE(20)
end program
