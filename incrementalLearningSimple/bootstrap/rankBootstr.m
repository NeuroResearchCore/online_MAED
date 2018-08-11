function [scoreSorted,indices] = rankBootstr(indices,scores,N)
mapObj = containers.Map('KeyType','int32','ValueType','double')
for j=1:N
    scorebootstr=scores(j,:);
    for i=1:size(scorebootstr,2)
        if isKey(mapObj,indices(j,i))
            mapObj(indices(j,i))=mapObj(indices(j,i))+scorebootstr(i);
        else
            mapObj(indices(j,i))=scorebootstr(i);
        end
    end
end
averageScore=cell2mat(values(mapObj))./N;
[scoreSorted, scoreOrder] = sort(averageScore,'descend'); 
indices=cell2mat(keys(mapObj));
indices=indices(:,scoreOrder);
end

