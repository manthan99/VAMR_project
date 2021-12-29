function [L,Locb] = ismembertol_Custom(A, B, tol)
% finds matching members in A and B with tolerance = tol
% L - logical vector which is 1 where element of A is a member of B, Locb: vector of indices in B for each element in A that is a member of B

    tol = 2;
    [L,Locb] = ismembertol(A, B, tol,'ByRows',true, 'DataScale', 1);
end


