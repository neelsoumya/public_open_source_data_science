function varargout = matlabpool(varargin)
    % matlabpool function since it is no longer supported
    % code from
    % https://uk.mathworks.com/matlabcentral/answers/235524-how-to-replace-matlapool-by-parpool-in-my-code
    % by Matt J https://uk.mathworks.com/matlabcentral/profile/authors/1440443-matt-j

     varargout={};
     arg1=varargin{1};
     switch arg1
       case 'open'
         parpool(varargin{2:end});
       case 'close'
         delete(gcp('nocreate'));
       case  'size'
         p = gcp('nocreate'); % If no pool, do not create new one.
        if isempty(p)
            poolsize = 0;
        else
            poolsize = p.NumWorkers;
        end
        varargout={poolsize};
    otherwise %number of workers
         if ~isnumeric(arg1)
           arg1=num2str(arg1);
         end
         parpool(arg1);
end