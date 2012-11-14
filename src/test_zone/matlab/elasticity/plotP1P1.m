function plotP1P1(n4e,c4n,x,lambda,mu,factor)
%%   SHOW  Plots two-dimensional solution
%    SHOW(n4e,c4n,x,lambda,mu,factor) plots the
%    strained mesh and visualizes the stresses in grey tones.
%
%    The variable AVS is previously determined by the function <avmatrix.m>.
%
%
%   See also FEM_LAME2D and AVMATRIX.

%    J. Alberty, C. Carstensen, S. A. Funken, and R. Klose  07-03-00
%    File <show.m> in $(HOME)/acfk/fem_lame2d/cooks/ and
%                     $(HOME)/acfk/fem_lame2d/lshape_p1/ and
%                     $(HOME)/acfk/fem_lame2d/lshape_q1/ and
%                     $(HOME)/acfk/fem_lame2d/hole/


%% Compute AvS
Sigma3 = zeros(size(n4e,1),4);
AreaOmega = zeros(size(c4n,1),1);
AvS = zeros(size(c4n,1),4);
for j = 1:size(n4e,1)
  area4e = computeArea4e(c4n,n4e);
  AreaOmega(n4e(j,:)) = AreaOmega(n4e(j,:)) +(area4e(j)*ones(1,3))';
  PhiGrad = [1,1,1;c4n(n4e(j,:),:)']\[zeros(1,2);eye(2)];
  U_Grad = x([1;1]*2*n4e(j,:)-[1;0]*[1,1,1])*PhiGrad;
  Sigma3(j,:) = reshape(lambda*trace(U_Grad)*eye(2) ...
      +2*mu*(U_Grad+U_Grad')/2,1,4);
  AvS(n4e(j,:),:) = AvS(n4e(j,:),:) +area4e(j)*[1;1;1]*Sigma3(j,:);
end;
AvS = AvS./(AreaOmega*[1,1,1,1]);

%% Plot the solution
for i=1:size(c4n,1)
 AvC(i)=(mu/(24*(mu+lambda)^2)+1/(8*mu))*(AvS(i,1)+...
        AvS(i,4))^2+1/(2*mu)*(AvS(2)^2-AvS(1)*AvS(4));
end
colormap(1-gray)
trisurf(n4e,factor*x(1:2:size(x,1))+c4n(:,1), ...
    factor*x(2:2:size(x,1))+c4n(:,2), ...
    zeros(size(c4n,1),1), AvC, 'facecolor','interp');
hold on
view(0,90)
hold off
colorbar('vert')
  



