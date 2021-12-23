function [L,Locb] = ismembertol_Custom(A, B, tol)
    [L,Locb] = ismembertol(A, B, tol,'ByRows',true);
end


