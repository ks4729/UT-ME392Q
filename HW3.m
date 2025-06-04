%{
%import data
QuerySpectrum = readmatrix("QuerySpectrum.xlsx");
SpectrumLibrary = readmatrix("SpectrumLibrary.xlsx");
%clean up data
SpectrumToAnalyze = QuerySpectrum(2, :);

%no exact match, started small and raised until one answer was produced by
%program
toleranceBand = 0.001;

for i = 1:height(SpectrumLibrary)
   for j = 1:length(SpectrumToAnalyze)
        if (SpectrumLibrary(i,j+7) - toleranceBand <= SpectrumToAnalyze(j)) && (SpectrumLibrary(i,j+7) + toleranceBand >= SpectrumToAnalyze(j))
            if j == length(SpectrumToAnalyze)
                disp(i);
            end
        else
            break
        end
   end    
end
%}

%{

Vectors = readmatrix("Homework3Problem2.xlsx");

%use population mean and std to center so we use tests like
%Kolmogorov–Smirnov which check against standard normal distribution (mean
%of 0, std of 1)

x1 = (Vectors(:,1) - mean(Vectors(:,1)))/std(Vectors(:,1));
x2 = (Vectors(:,2) - mean(Vectors(:,2)))/std(Vectors(:,2));
x3 = (Vectors(:,3) - mean(Vectors(:,3)))/std(Vectors(:,3));

disp(kstest(x1))
disp(kstest(x2))
disp(kstest(x3))


figure(1)
histogram(x1)
figure(2)
histogram(x2)
figure(3)
histogram(x3)
%}

x = [20.85 20.5 20.1 20.2 19.8 19.9 20.5 20.2 20.3];
nu = mean(x);
sigma = std(x);
testx = (x-nu)/sigma;
disp((nu-20)/(sigma/sqrt(9)));

