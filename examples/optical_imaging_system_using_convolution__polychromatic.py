import diffractsim
diffractsim.set_backend("CUDA")

from diffractsim import PolychromaticField, ApertureFromImage, CircularAperture, cf, nm, mm, cm


size_factor = 1.0
F = PolychromaticField(
    spectrum=2 * cf.illuminant_d65, extent_x=size_factor* 1.5 * mm, extent_y=size_factor* 1.5 * mm, Nx=2048, Ny=2048,
    spectrum_size = 180, spectrum_divisions = 30
)


F.add(ApertureFromImage( "./apertures/horse.png",  image_size=(size_factor*1.0 * mm, size_factor*1.0 * mm), simulation = F))

rgb = F.get_colors_at_image_plane(pupil = CircularAperture(radius = 6*mm) ,  zi = 50*cm, z0 = 50*cm)

F.plot_colors(rgb, figsize=(5, 5), xlim=[-size_factor*0.4*mm,size_factor*0.4*mm], ylim=[-size_factor*0.4*mm,size_factor*0.4*mm])
