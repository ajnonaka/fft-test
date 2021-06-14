#include <AMReX.H>
#include <AMReX_MultiFab.H>
#include <AMReX_LayoutData.H>
#include <AMReX_ParmParse.H>
#include <AMReX_GpuComplex.H>

#include <fftw3.h>
#include <fftw3-mpi.h>

using namespace amrex;

int main (int argc, char* argv[])
{
    amrex::Initialize(argc, argv); {

    fftw_mpi_init();

    BL_PROFILE("main");

    Geometry geom;
    BoxArray ba;
    DistributionMapping dm;
    IntVect nghost;
    {
        ParmParse pp;
        IntVect n_cell;
        pp.get("n_cell", n_cell);

        Box domain(IntVect(0),n_cell-IntVect(1));
        RealBox rb({0.,0.,0.},{1.,1.,1.});
        Array<int,3> is_periodic{1,1,1};
        geom.define(domain, rb, CoordSys::cartesian, is_periodic);

        // make the domain one box
        ba.define(domain);
        dm.define(ba);
    }

    // the real-space data (no ghost cells)
    MultiFab real_field(ba,dm,1,0);

    // initialize data to random numbers
    for (MFIter mfi(real_field); mfi.isValid(); ++mfi) {
        Array4<Real> const& fab = real_field.array(mfi);
        amrex::ParallelFor(mfi.fabbox(),
        [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
        {
            fab(i,j,k) = amrex::Random();
        });
    }

    using FFTplan = fftw_plan;
    using FFTcomplex = fftw_complex;

    // contain to store FFT - note it is shrunk by "half" in x
    Vector<std::unique_ptr<BaseFab<GpuComplex<Real> > > > spectral_field;

    Vector<FFTplan> forward_plan;
    
    for (MFIter mfi(real_field); mfi.isValid(); ++mfi) {
        
        // grab a single box including ghost cell range
        Box realspace_bx = mfi.fabbox();

        // size of box including ghost cell range
        IntVect fft_size = realspace_bx.length(); // This will be different for hybrid FFT

        // this is the size of the box, except the 0th component is 'halved plus 1'
        IntVect spectral_bx_size = fft_size;
        spectral_bx_size[0] = fft_size[0]/2 + 1;

        // spectral box
        Box spectral_bx = Box(IntVect(0), spectral_bx_size - IntVect(1));
        
        spectral_field.emplace_back(new BaseFab<GpuComplex<Real> >(spectral_bx,1,
                                                                   The_Device_Arena()));
        spectral_field.back()->setVal<RunOn::Device>(0.0); // touch the memory

        FFTplan fplan;        
        fplan = fftw_plan_dft_r2c_3d(fft_size[2], fft_size[1], fft_size[0],
                                     real_field[mfi].dataPtr(),
                                     reinterpret_cast<FFTcomplex*>
                                         (spectral_field.back()->dataPtr()),
                                     FFTW_ESTIMATE);

        forward_plan.push_back(fplan);
    }

    ParallelDescriptor::Barrier();

    // ForwardTransform
    for (MFIter mfi(real_field); mfi.isValid(); ++mfi) {
        int i = mfi.LocalIndex();
        fftw_execute(forward_plan[i]);
    }

    // storage for the answer (on one grid)
    MultiFab variables_dft_real(ba,dm,1,nghost);
    MultiFab variables_dft_imag(ba,dm,1,nghost);
    
    // copy data to a full-sized MultiFab
    // this involves copying the complex conjugate from the half-sized field
    // into the appropriate place in the full MultiFab
    for (MFIter mfi(variables_dft_real); mfi.isValid(); ++mfi) {

        Array4< GpuComplex<Real> > spectral = (*spectral_field[0]).array();

        Array4<Real> const& realpart = variables_dft_real.array(mfi);
        Array4<Real> const& imagpart = variables_dft_imag.array(mfi);
        
        Box bx = mfi.fabbox();
        
        amrex::ParallelFor(bx,
        [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
        {
            if (i <= bx.length(0)/2) {
                // copy value
                realpart(i,j,k) = spectral(i,j,k).real();
                imagpart(i,j,k) = spectral(i,j,k).imag();
            } else {
                // copy complex conjugate
                int iloc = bx.length(0)-i;
                int jloc = (j == 0) ? 0 : bx.length(1)-j;
                int kloc = (k == 0) ? 0 : bx.length(2)-k;

                realpart(i,j,k) =  spectral(iloc,jloc,kloc).real();
                imagpart(i,j,k) = -spectral(iloc,jloc,kloc).imag();
            }                
        });     
        
        amrex::ParallelFor(bx,
        [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
        {
            std::cout << "HACKFFT " << i << " " << j << " " << k << " "
                      << realpart(i,j,k) << " + " << imagpart(i,j,k,0) << "i" << std::endl;
        });
    }
    
    // destroy fft plan
    for (int i = 0; i < forward_plan.size(); ++i) {
        fftw_destroy_plan(forward_plan[i]);
    }

    fftw_mpi_cleanup();

    } amrex::Finalize();
}
