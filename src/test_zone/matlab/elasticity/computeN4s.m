function n4s = computeN4s(n4e)
%% computeN4s - Nodes for sides.
%   computeN4s(n4e) returns a matrix in which each row corresponds to one side 
%               of the decomposition. The side numbering is the same as in
%               e4s, s4n, s4e, length4s, mp4s, normal4s and tangent4s. Each
%               row consists of the numbers of the end nodes of the 
%               corresponding side. n4e is as specified in the 
%               documentation.
%
%   See also: computeE4s, computeS4n, computeS4e, computeLength4s,
%             computeMid4s, computeNormal4s, computeTangent4s

    if isempty(n4e)
        n4s = [];
        return;
    end

    %% Compute n4s.
    % Gather a list of all sides including duplicates (occurring for inner
    % sides), then sort each row and make sure the rows are unique, thus
    % eliminating duplicates.
    allSides = [n4e(:,[1 2]); n4e(:,[2 3]); n4e(:,[3 1])];
    % Eliminate duplicates, remember original index of remaining rows.
    [b,ind] = unique(sort(allSides,2),'rows','first');
    n4s = allSides(sort(ind),:);
end