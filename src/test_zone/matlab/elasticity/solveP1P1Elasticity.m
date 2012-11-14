function [x,nrDof,A,b] = solveP1P1Elasticity(f,g,u4Db,c4n,n4e,n4sDb,n4sNb,mu,lambda)
    %% Initialisation
    A = sparse(2*size(c4n,1),2*size(c4n,1)); 
    b = zeros(2*size(c4n,1),1);
    nrNodes = size(c4n,1);
    DirichletNodes = unique(n4sDb);
    dof = setdiff(1:nrNodes,DirichletNodes); % free nodes to be approximated
    nrDof = length(dof);

    % Assembly
    for j = 1:size(n4e,1)
        I = 2*n4e(j,[1,1,2,2,3,3]) -[1,0,1,0,1,0]; 
        A(I,I) = A(I,I) +stima3(c4n(n4e(j,:),:),lambda,mu);   
    end

    % Volume forces
    for j = 1:size(n4e,1)
        I = 2*n4e(j,[1,1,2,2,3,3]) -[1,0,1,0,1,0];
        fs = f(sum(c4n(n4e(j,:),:))/3)';
        b(I) = b(I) +det([1,1,1;c4n(n4e(j,:),:)'])*[fs;fs;fs]/6;
    end

    % Neumann conditions
    if ~isempty(n4sNb)
        n = (c4n(n4sNb(:,2),:) -c4n(n4sNb(:,1),:))*[0,-1;1,0];
        for j = 1:size(n4sNb,1);
            I = 2*n4sNb(j,[1,1,2,2]) -[1,0,1,0];
            gm = g(sum(c4n(n4sNb(j,:),:))/2, n(j,:)/norm(n(j,:)))';
            b(I) = b(I) +norm(n(j,:))*[gm;gm]/2;
        end
    end

    % Dirichlet conditions (Note: For this particular example, the Lame
    % constants LAMBDA and MU are also needed for the Dirichlet boundary
    % condition.)
    [W,M] = u4Db(c4n(DirichletNodes,:),lambda,mu);
    B = sparse(size(W,1),2*size(c4n,1));
    for k = 0:1
        for l = 0:1
            B(1+l:2:size(M,1),2*DirichletNodes-1+k) = diag(M(1+l:2:size(M,1),1+k));
        end
    end
    mask = find(sum(abs(B)'));
    A = [A, B(mask,:)'; B(mask,:), sparse(length(mask),length(mask))];
    b = [b;W(mask,:)];

    % Calculating the solution
    x = A \ b;
    x = x(1:2*size(c4n,1)); %Remove Lagrange multipliers
end


%% Additional functions
function stima3=stima3(vertices,lambda,mu)
    %STIMA3   Computes element stiffness matrix for triangles.
    %   M = STIMA3(X,LAMBDA,MU) computes element stiffness matrix for
    %   triangles. The coordinates of the vertices are stored in X. LAMBDA
    %   and MU are the Lame constants.
    %
    %   This routine should not be modified.
    %
    %
    %   See also FEM_LAME2D and STIMA4.

    %    J. Alberty, C. Carstensen and S. A. Funken  07-03-00
    %    File <stima3.m> in $(HOME)/acfk/fem_lame2d/cooks/ and
    %                       $(HOME)/acfk/fem_lame2d/lshape_p1/ and
    %                       $(HOME)/acfk/fem_lame2d/lshape_q1/ and
    %                       $(HOME)/acfk/fem_lame2d/hole/

    PhiGrad = [1,1,1;vertices']\[zeros(1,2);eye(2)];
    R = zeros(3,6);
    R([1,3],[1,3,5]) = PhiGrad';
    R([3,2],[2,4,6]) = PhiGrad';
    C = mu*[2,0,0;0,2,0;0,0,1] +lambda*[1,1,0;1,1,0;0,0,0];
    stima3 = det([1,1,1;vertices'])/2*R'*C*R;
end