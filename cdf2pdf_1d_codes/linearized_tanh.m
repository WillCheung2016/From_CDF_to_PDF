function y = linearized_tanh(x)
    y = min(max(x, -1), 1);
end