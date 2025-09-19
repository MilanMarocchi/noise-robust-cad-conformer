%This function will join all Dia1, Dia2 and Dia3 together,
%1 second of end and start will be noted as 'poor' quality
clear
outdir = '/home/matthew-fynn/Desktop/TH_data_joined_indicies_SpikeRemoval/'; %where concatenated files are to be saved
mkdir(outdir)
addpath('/home/matthew-fynn/Desktop/TH_alldata') %where recordinsg are saved
fs_new = 2000;           % Sampling frequency (Hz)

%get a list of all filenames and exclude Bell, Lie, 10s
folder_path = '/home/matthew-fynn/Desktop/TH_alldata'; 
files = dir(folder_path); % Get list of all files and folders
file_names = {files.name};

filtered_files = file_names;
order=[1,2,3,4,5,6,7]; %seven heart mics - order could change based on connections while recording
%get list of subjects - can change this later
data = readtable('REFERENCE_ALLROUNDS.csv'); %subject and label information
subs = table2cell(data(:,1));
for s = 1:length(subs)
    %extract all of the files from this subject
    sub = subs{s};
    Sub_files = filtered_files(contains(filtered_files, sub));
    
    data_concat = [];
    len_array(1) = 0; %so i can access it in the for loop after
    bad_ind = [];
    for f = 1:length(Sub_files)
        file = Sub_files{f};

        [data_wav,fs] = audioread(file);
        data_wav = resample(data_wav,fs_new,fs);

        ind_medPow_HM_chan1 = segment_signal_noise_multi_MedPow(data_wav(:,1), 1.5, fs_new, 2.5);
        ind_medPow_HM_chan2 = segment_signal_noise_multi_MedPow(data_wav(:,2), 1.5, fs_new, 2.5);
        ind_medPow_HM_chan3 = segment_signal_noise_multi_MedPow(data_wav(:,3), 1.5, fs_new, 2.5);
        ind_medPow_HM_chan4 = segment_signal_noise_multi_MedPow(data_wav(:,4), 1.5, fs_new, 2.5);
        % ind_medPow_HM_chan5 = segment_signal_noise_multi_MedPow(data_wav(:,5), 1.5, fs_new, 2.5);
        % ind_medPow_HM_chan6 = segment_signal_noise_multi_MedPow(data_wav(:,6), 1.5, fs_new, 2.5);

        ind_medPow_NM = segment_signal_noise_multi_MedPow(data_wav(:,4+8),0.25,fs_new, 2.5); %NM is HM + 8

        %spike Removal for each channel
        for chan = 1:7
            col = order(chan);
            temp_chan = data_wav(:,col);
            temp_chan(fs_new-200:end-1000) = schmidt_spike_removal(temp_chan(fs_new-200:end-1000),fs_new); %Open source function - schmidt spike removal
            data_wav(:,col) = temp_chan;
        end
        
        
        data_concat = [data_concat; data_wav];        
        bad_ind_NM = ind_medPow_NM + sum(len_array); %as we are concatenating different .wav files together

        bad_ind_HM_chan1 = ind_medPow_HM_chan1 + sum(len_array);
        bad_ind_HM_chan2 = ind_medPow_HM_chan2 + sum(len_array);
        bad_ind_HM_chan3 = ind_medPow_HM_chan3 + sum(len_array);
        bad_ind_HM_chan4 = ind_medPow_HM_chan4 + sum(len_array);
        % bad_ind_HM_chan5 = ind_medPow_HM_chan5 + sum(len_array);
        % bad_ind_HM_chan6 = ind_medPow_HM_chan6 + sum(len_array);

        len_array(f+1) = length(data_wav);
        bad_ind = [bad_ind;bad_ind_NM;bad_ind_HM_chan1;bad_ind_HM_chan2;bad_ind_HM_chan3;bad_ind_HM_chan4];

        clearvars data_wav
    end
    %calculate start and end of file indicies here (count them as bad indices)

    chop = floor(1*fs_new)+1; %exclude first and last second of each recording
    x=1;
    for i = 1:length(len_array)-1
        good_ind(x,1) = chop + sum(len_array(1:i)); x=x+1;
        good_ind(x,1) = sum(len_array(1:i+1)) - chop; x=x+1;
    end
    
    %FIRST WE HAVE TO MERGE THE BAD IND (could get same indicies from HM and BNM in same recording or across different channels
    bad_ind_s = sortrows(bad_ind);
    merged_bad_ind = [];
    current_range = bad_ind_s(1,:);
    
    for i = 2:size(bad_ind_s, 1)
        this_range = bad_ind_s(i,:);
        
        if this_range(1) <= current_range(2) + 1
            % Overlapping or adjacent ranges, merge them
            current_range(2) = max(current_range(2), this_range(2));
        else
            % No overlap, store the current range
            merged_bad_ind = [merged_bad_ind; current_range];
            current_range = this_range;
        end
    end
    
    % Don't forget to add the last range
    merged_bad_ind = [merged_bad_ind; current_range];

    %NOW WE HAVE TO CHANGE GOOD IND
    % Step 2: Subtract bad intervals from good intervals
    new_good_ind = [];
    
    for i = 1:2:length(good_ind)
        good_start = good_ind(i);
        good_end = good_ind(i+1);
        
        % Find bad intervals that overlap with this good interval
        overlapping = merged_bad_ind(merged_bad_ind(:,2) >= good_start & merged_bad_ind(:,1) <= good_end, :);
        
        if isempty(overlapping)
            % No bad parts, keep the whole good interval
            new_good_ind = [new_good_ind; good_start; good_end];
        else
            % Split the good interval around bad parts
            current_start = good_start;
            for j = 1:size(overlapping,1)
                bad_start = max(overlapping(j,1), good_start); % Clip to good interval
                bad_end = min(overlapping(j,2), good_end);
                
                if current_start < bad_start
                    new_good_ind = [new_good_ind; current_start; bad_start - 1];
                end
                current_start = bad_end + 1;
            end
            
            % Add any remaining good part after last bad interval
            if current_start <= good_end
                new_good_ind = [new_good_ind; current_start; good_end];
            end
        end
    end
    
    % Optional: Make sure it's a column vector
    new_good_ind = new_good_ind(:);

    fname_save = [outdir,sub];
    % audiowrite([fname_save,'.wav'],data_concat,fs_new, 'BitsPerSample',32);
    ind_cell{s,1} = sub;
    ind_cell{s,2} = new_good_ind;
    clearvars data_* len_array good_ind good_ind new* bad* merged*
end

ind_table = cell2table(ind_cell);
ind_table.Properties.VariableNames{'ind_cell1'} = 'subject';
writetable(ind_table,'Ind_Table_MedPow_1234HM_4NM.csv');

