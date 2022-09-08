function y = mytanh(x)
    %x = gather(x);
    y = (1 - exp(-2*x)) ./ (1 + exp(-2*x));
    %sigm = gpuArray(sigm);
end