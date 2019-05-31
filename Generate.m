clear all
params = zeros(10000,5);
t =  0.005e-6;
for i=1:1:10000
    w = rand*(10^-4);
    e = rand*(10);
    h = rand*(10^-4);   
    s = rand*(10^-4);
    while e<1
        e = rand*(10);
    end
    if (w/s < 0.01 || w/s >100)
        w = 0.6e-4;
        s = 0.2e-4;
    end
    if (s/h>20 || s/h<0.05)
        h = 0.635e-4;
        s = 0.2e-4;
    end
    if (t/s > 0.1)
        s = 0.2e-4;
    end
    tx=rfckt.cpw('ConductorWidth',w,'EpsilonR',e,'Height',h,'SlotWidth',s,'Thickness',t);
    analyze(tx,1e9);
    z0 = getz0(tx);
    params(i,:) = [w,h,s,e,z0];
end

params(:,1:3) = params(:,1:3)*(10^5);
