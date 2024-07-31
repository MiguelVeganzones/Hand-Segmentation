<<<<<<< HEAD
% 24 / 12 / 2021 
% get the ground truth of every image.
% images are stored in a different root folder,
% with same subfolders and structure % 

addpath("egohands/")
videos = getMetaBy();
foldername = "./annotations/_GROUND_TRUTH_DISJUNCT_HANDS";
mkdir(foldername);
SE = strel('disk', 2, 8);
display(SE);
binarize_threshold = 0.35;

for i = 1 : length(videos)
    videos_i = videos(i);
    subfoldername = videos_i.video_id;
    mkdir(foldername + "/" + subfoldername);
    for j = 1:length(videos_i.labelled_frames)
        mr = getSegmentationMask(videos_i, j, 'my_right');
        ml = getSegmentationMask(videos_i, j, 'my_left');
        yr = getSegmentationMask(videos_i, j, 'your_right');
        yl = getSegmentationMask(videos_i, j, 'your_left');

        mr_e = imdilate(mr, SE);
        ml_e = imdilate(ml, SE);
        yr_e = imdilate(yr, SE);
        yl_e = imdilate(yl, SE);

        intersection = mr_e .*( ml_e + yr_e + yl_e) + ml_e .* (yr_e + yl_e) + yr_e .* yl_e; 
        dilated_intersection = imbinarize(imdilate(intersection, SE), binarize_threshold);
        seg_mask = imbinarize(imbinarize(mr + ml + yr + yl, binarize_threshold) - dilated_intersection, binarize_threshold);

        imwrite(seg_mask,
        foldername + "\" + subfoldername +
        "\frame_" + num2str(videos_i.labelled_frames(j).frame_num,'%04.f')+ ".jpg", 'jpg');
    end
end

display('Done');
=======
% 24 / 12 / 2021 
% get the ground truth of every image.
% images are stored in a different root folder,
% with same subfolders and structure % 

addpath("egohands/")
videos = getMetaBy();
foldername = "./annotations/_GROUND_TRUTH_DISJUNCT_HANDS";
mkdir(foldername);
SE = strel('disk', 2, 8);
display(SE);
binarize_threshold = 0.35;

for i = 1 : length(videos) 
    videos_i = videos(i);
    subfoldername = videos_i.video_id;
    mkdir(foldername + "/" + subfoldername);
    for j = 1:length(videos_i.labelled_frames)
        mr = getSegmentationMask(videos_i, j, 'my_right');
        ml = getSegmentationMask(videos_i, j, 'my_left');
        yr = getSegmentationMask(videos_i, j, 'your_right');
        yl = getSegmentationMask(videos_i, j, 'your_left');
        
        mr_e = imdilate(mr, SE);
        ml_e = imdilate(ml, SE);
        yr_e = imdilate(yr, SE);
        yl_e = imdilate(yl, SE);
        
        intersection = mr_e .*( ml_e + yr_e + yl_e) + ml_e .* (yr_e + yl_e) + yr_e .* yl_e; 
        dilated_intersection = imbinarize(imdilate(intersection, SE), binarize_threshold);
        seg_mask = imbinarize(imbinarize(mr + ml + yr + yl, binarize_threshold) - dilated_intersection, binarize_threshold);
               
        imwrite(seg_mask, ...
        foldername + "\" + subfoldername + ... 
        "\frame_" + num2str(videos_i.labelled_frames(j).frame_num,'%04.f')+ ".jpg", 'jpg');   
    end
end

display('Done');
>>>>>>> refs/remotes/upstream/main
