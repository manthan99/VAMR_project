function [L,Locb] = ismembertol_Custom(A, B, tol)
    tol = 2;
    [L,Locb] = ismembertol(A, B, tol,'ByRows',true, 'DataScale', 1);
end


