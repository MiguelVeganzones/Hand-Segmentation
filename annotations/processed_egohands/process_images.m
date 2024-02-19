% % 24 / 12 / 2021 % get the ground truth of every image.%
    images are stored in a different root folder,
    % with same subfolders and structure % % videos = getMetaBy();

foldername = ".\_GROUND_TRUTH_DISJUNCT_HANDS_2";
weight_map_foldername = ".\_WEIGHT_MAPS_08_03_22";
mkdir(foldername);
SE = strel('disk', 2, 8);
display(SE);

for
  i = 1 : length(videos) videos_i = videos(i);
subfoldername = videos_i.video_id;
    mkdir(foldername + "\" + subfoldername);
    for j = 1:length(videos_i.labelled_frames)
        mr = getSegmentationMask(videos_i, j, 'my_right');
        ml = getSegmentationMask(videos_i, j, 'my_left');
        yr = getSegmentationMask(videos_i, j, 'your_right');
        yl = getSegmentationMask(videos_i, j, 'your_left');
        
        mr_e = imdilate(mr, SE);
        ml_e = imdilate(ml, SE);
        yr_e = imdilate(yr, SE);
        yl_e = imdilate(yl, SE);
        
        intersection = imbinarize(mr_e .*( ml_e + yr_e + yl_e) + ...
                                  ml_e .* (yr_e + yl_e) + yr_e .* yl_e, 0); 
        imshow(intersection);
        dilated_intersection = imdilate(intersection, SE);
        
        seg_mask = imbinarize(mr + ml + yr + yl, 0.5) - dilated_intersection;
        
         seg_mask = imerode(getSegmentationMask(videos_i, j, 'my_right'), SE)...
                  + imerode(getSegmentationMask(videos_i, j, 'my_left'), SE)...
                  + imerode(getSegmentationMask(videos_i, j, 'your_right'), SE)...
                  + imerode(getSegmentationMask(videos_i, j, 'your_left'), SE);
         %'all' for all hands, Can also be:... 
         % my_right, yours, your_left....
        
%         imwrite(getSegmentationMask(videos_i, j, 'all');
        
%         if sum(any(intersection)) > 50
%             imshow(mr + ml + yr + yl)
%             uiwait
%             imshow(mr_e + ml_e + yr_e + yl_e)
%             uiwait
%             imshow(intersection)
%             uiwait
%             imshow(dilated_intersection)
%             uiwait
%             imshow(seg_mask)
%             uiwait
%         end
         
        imwrite(seg_mask, ...
        foldername + "\" + subfoldername + ... 
        "\frame_" + num2str(videos_i.labelled_frames(j).frame_num,'%04.f')+ ".jpg", 'jpg');   
    end
end

display('Done');
