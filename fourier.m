%fourier transform of the image

function [I_mag]=fourier(I_in)

fft= fft2(I_in);
I_mag=abs(fft); %frequency spectrum

end

