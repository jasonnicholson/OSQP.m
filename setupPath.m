function setupPath(shouldSetup)
% setupPath adds or removes the src folder from the MATLAB path
  arguments (Input)
    shouldSetup (1,1) logical = true;
  end

  repoRoot = string(fileparts(mfilename("fullpath")));
  src = fullfile(repoRoot, "src");
  qdldlSrc = fullfile(repoRoot, "qdldl", "src");
  p = [src; qdldlSrc];

  p = convertStringsToChars(p);

  if shouldSetup
    addpath(p{:});
  else
    rmpath(p{:});
  end
end