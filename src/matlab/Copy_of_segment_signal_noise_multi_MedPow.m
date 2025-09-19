function ind_flag = segment_signal_noise_multi_MedPow(signal, frame_len_sec, fs_new, threshold)
    len_signal = length(signal);
    frame_len_sample = frame_len_sec * fs_new;
    no_frames = floor(len_signal / frame_len_sample);
    
    En = zeros(1, no_frames);
    
    for i = 1:no_frames
        sig = signal((i-1)*frame_len_sample+1 : i*frame_len_sample);
        En(i) = sum(sig.^2);
    end
    
    med_val = median(En(2:end-1));

    j = 1;
    ind_flag = [];
    
    for i = 1:length(En)
        if En(i) > threshold * med_val
            ind_flag(j,1) = (i-1) * frame_len_sample + 1;
            ind_flag(j,2) = i * frame_len_sample;
            j = j + 1;
        end
    end

end

