function area4e = computeArea4e(c4n,n4e)
%% computeArea4e - Area for elements.
%   computeArea4e(c4n, n4e) computes the area of each element of a
%                       decomposition where c4n, n4e are as specified in
%                       the documentation.
%
%   See also: computeArea4n

    if isempty(n4e)
        area4e = zeros(0,1);
        return;
    end
    
    %% Compute area4e.
    % Get the x- and y-coordinates for each node of each element and
    % compute the area of all elements simulateously.
    x1 = c4n(n4e(:,1),1);
    x2 = c4n(n4e(:,2),1);
    x3 = c4n(n4e(:,3),1);
    y1 = c4n(n4e(:,1),2);
    y2 = c4n(n4e(:,2),2);
    y3 = c4n(n4e(:,3),2);
    
    area4e = ( x1.*(y2 - y3) + x2.*(y3 - y1) + x3.*(y1 - y2) )/2;
    
end
