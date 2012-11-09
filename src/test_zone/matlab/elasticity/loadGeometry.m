function [c4n n4e n4sDb n4sNb] = loadGeometry(name, OPTRefinementLevel)
%% loadGeometry - load data for a mesh.
% [c4n n4e n4sDb n4sNb] = loadGeometry('name') loads the data structures for
%   the mesh named 'name'. Optionally, the second parameter will
%   cause the mesh to be refined a given number of times using the uniform
%   red strategy.
% Example:
% [c4n n4e n4sDb n4sNb] = loadGeometry('LShape',2) loads the
%   mesh called 'LShape' and refines it two times.

    %% Load the geometry data.
    c4n = load([name,'_c4n.dat']);
    n4e = load([name,'_n4e.dat']);
    n4sDb = load([name,'_n4sDb.dat']);
    n4sNb = load([name,'_n4sNb.dat']);
    
    %% Initial refinement.
    if nargin < 2
        OPTRefinementLevel = 0;
    end
    for i=1:OPTRefinementLevel
        [c4n,n4e,n4sDb,n4sNb] = refineUniformRed(c4n,n4e,n4sDb,n4sNb);
    end
end