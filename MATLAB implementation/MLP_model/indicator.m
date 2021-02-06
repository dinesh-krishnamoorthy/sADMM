function ind = indicator(y,nc)
for i = 1:nc
    if(y==i)
        ind(i) = 1;
    else
        ind(i) = 0;
    end
end
ind = ind';