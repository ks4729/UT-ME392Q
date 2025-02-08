load("HW2Problem1(1).mat");

frequency = 100; %Hz
nyquistFrequency = frequency/2;
FFTx = fft(x, 2^11); %2048 samples
frequencySamples = 0:nyquistFrequency*(2^-10):nyquistFrequency;
powerSpectralDensity = abs(FFTx);
relevantPSD = powerSpectralDensity(1:1+2^10); %isolate the first half

[Pxx,w]=pyulear(x,100,2048,"onesided");
frequencySamplesyulewalker = 0:(nyquistFrequency)*(1/(length(Pxx)-1)):(nyquistFrequency);

figure(1)
plot(t,x)
xlabel('Time [s]')
ylabel('Signal value')

figure(2)
tiledlayout(1,2)
nexttile
plot(frequencySamples,relevantPSD);
xlabel('Frequency [Hz]')
ylabel('Spectral intensity for a given frequency')
nexttile
plot(frequencySamplesyulewalker,Pxx);
xlabel('Frequency [Hz]')
ylabel('Y-W Spectral intensity for a given frequency')