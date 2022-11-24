%SOM

clear all
close all
%
% seme del random generator
% rng(123456789)
  rng(546465164)
 %carica file con i dati che si chiamano data050, data150 etc
 
%load('./Up/txt_tracce_up.txt','-ascii');

load('data_up_Velocimeter_8s.txt','-ascii');
%scegli i dati che vuoi, e li chiamo x2
% Matrice di dati (nel nostro caso sono 151 punti per ogni evento)
X0=data_up_Velocimeter_8s';
%X1=txt_tracce_down';
clear X2;
c=200;
pre=20;
post=5
clear X2;
%X2=[X0(c-pre:c+post,:),X1(c-pre:c+post,:)];
X2=X0(c-pre:c+post,:);

ss=size(X2)
cc=floor(ss(1)-post);
X3=X2-mean(X2(1:(cc-5),:),1);

X=normalize(X3,'scale');
%X=X3-std(X3(1:cc-5,:));

 t=1:1:size(X(:,1),1);

% a=1269
%  b=1956;
a=ss(2)
%{
figure
 for i=1:1:a
 hold on
 plot(t,X(:,i),'o-')
 ylabel('up')
 end
%}

%  
% figure
%  for i=a+1:1:b
%  hold on
%  plot(t,X(:,i),'o-')
%  ylabel('down')
%  end
 
 
 %% RE-SIZE
 clear medie;
 j=0;
 for passo=1:1:post
     j=j+1;
 medie(j,:)= mean(X(passo:(cc+passo-1),:),1);
 end
 
 %{
 figure
 plot(medie(:,1:a),'ro-')
 hold on
 %}

%   plot(medie(:,a+1:b),'ko-')
 
% 
%   
%   
%   clear mas;
%  j=0;
%  for passo=1:5:11
%      j=j+1;
%  mas(j,:)= max(X(passo:(cc+passo),:));
%  end
%  
%  figure
%  plot(mas(:,1:a),'ro-')
%  hold on
%   plot(mas(:,a+1:b),'ko-')
%  
%   
%   
%   
%   clear mi;
%  j=0;
%  for passo=1:3:11
%      j=j+1;
%  mi(j,:)= min(X(passo:(cc+passo-1),:));
%  end
%  
%  figure
%  plot(mi(:,1:a),'ro-')
%  hold on
%   plot(mi(:,a+1:b),'ko-')
%  
  clear provamedie 
   provamedie=medie(:,:)-medie(1,:);
 
   %{ 
   figure
   plot(provamedie,'ko-')
   hold on
   plot(provamedie(:,1:a),'ro-');
   %}
%    
% for j2=1:size(X,2)
%     Ma2(j2)=max(X(:,j2));
% end
% for j2=1:size(X2,2)
%     Mi2(j2)=min(X2(:,j2));
% end
% 
% %si guardano le quantità calcolate
% figure(1)
%  scatter(Mi2,Ma2)
% xlabel('minVLP ')
%  ylabel('maxVLP')



size(X)
clear score
clear coeff
clear tsquared
clear explained
% ESTRAGGO COMPONENTI PCA
[coeff,score,latent,tsquared,explained] = pca(X');
explained;
%{
figure
size(score)
%scatter3(score(:,1),score(:,2),score(:,3))
axis equal
 xlabel('1st Principal Component')
ylabel('2nd Principal Component')
% zlabel('3rd Principal Component')
 scatter(score(1:a,1),score(1:a,2),'ro')
 hold on
%   scatter(score(a+1:b,1),score(a+1:b,2),'ko')
 hold on
 
 xlabel('1st Principal Component')
ylabel('2nd Principal Component')

figure
scatter3(score(1:a,1),score(1:a,2),score(1:a,3),'ko')
xlabel('1st Principal Component')
ylabel('2nd Principal Component')
zlabel('3rd Principal Component')
%}
rng(123456)

ind=4;


%% SI CREA LA SOM con i parametri di adderstramento .. svegliere i parametri !
 net = selforgmap([5 5],1000,3)
 
 net.trainParam.epochs = 3000;
 
 % in P si mettono i dati da dare in ingresso alla SOM,
 
 % cioe' un certo numero (ind, da scegliere) di componenti principali e i max e min
clear P;
 P=score(:,1:ind)';
 nn=size(provamedie,1)
  P(ind+1:ind+nn-1,:)=provamedie(2:nn,:);
%   P(ind+4:ind+4,:)=Mi2;
  
 %P(ind+5:ind+5,:)=(Ma-Mi);
 %P(ind+6:ind+6,:)=(Ma2-Mi2);

P=normalize(P);
 [net,tr] = train(net,P)
 

outputs = net(P);

plotsomhits(net,P);



y = net(P);
     classes = vec2ind(y);
     


%{
figure
nodo=7;
kk=find(y(nodo,:)==1);
 hold on
% plot(t,X2(:,:),'-.')
 plot(t,X2(:,kk(kk>0)),'bo-')  %nodo 7
 ylabel(' nodo 7')
%}

% 
% % figure

plotsomnd(net)

clear k
clear k1;
clear k2;

clear knodo;
clear nodo;
%{
nodo=6;
knodo=find(y(nodo,:)==1);
k3=knodo
figure
for i=1:1:size(k3,2)
    hold on
    if k3(i)>0
    plot(X(:,k3(i)),'-','LineWidth',2) 
    end
    txt = ['\bf nodo ',num2str(nodo)];
end
%}

%k2=find(y(9,:)==1)-(a1)
%
%SALVO IL NODO CUI APPARTIENE OGNI DATO
 cl=(classes');
save('4classes_down.txt','cl','-ascii');
 
 plotsomnd(net)
   savefig('cok5Distanze_tranodiSOM.fig')
   
  % save('cok5net.mat','net','P')
   
   plotsomhits(net,P)
   savefig('cok4_popolosita_hits.fig')
   
   n=5;
   
   

figura_nodi = figure;
title('som SOLO DOWN, giallo=media ')
subplot(n,n,1)
for j=1:(n*n)
    pos=(n*(n-1)+1)-(n*(idivide(int32(j-1),int32(n),'floor')))+mod((j-1),n);
     clear k3 ka1 ka2 ka3
k3=find(cl(:,:)==j);
  subplot(n,n,double(pos)) 
 STACKJ(j,:)=mean(X(:,k3),2);
 hold on
  plot(X(:,k3),'-','LineWidth',0.1) 
  plot(mean(X(:,k3),2),'k','LineWidth',2.0) 
 title(j)
end
% saveas(figura_nodi, "nodi.jpg")
% % 
% % subplot(n,n,1)
% % for j=1:(n*n)
% %     pos=(n*(n-1)+1)-(n*(idivide(int32(j-1),int32(n),'floor')))+mod((j-1),n);
% %      clear k3 ka1 ka2 ka3
% % k3=find(cl(1:a,1)==j);
% % %k1=find(cl(1+a:b,1)==j)+a;
% %   subplot(n,n,double(pos)) 
% % % STACKJ(j,:)=mean(X(:,k3),2);
% %  hold on
% %   plot(X(:,k3),'b','LineWidth',0.1) 
% %    plot(mean(X(:,k3),2),'g','LineWidth',2.0)
% %  %  plot(mean(X(:,k1),2),'m','LineWidth',2.0)
% % %    plot(mean(X(:,k3),2),'k','LineWidth',2.0) 
% %   plot(STACKJ(j,:),'y','LineWidth',2.0) 
% %  title(j)
% % end
title('SOM SOLO DOWN  ')
 %save('cok5STACK.mat','STACKJ','-v7.3')
 
 %%salvo dati in formato ascii
 %save('dati_ascii.txt','X','-ascii')

 % roba per figure che non interessa ora
 %set(findobj(get(gca, 'Children'), 'Type', 'Text'), 'Color', 'k');
% patches = findobj(get(gca, 'Children'), 'Type', 'Patch');  % All the patch objects
%set(patches(1:25), 'FaceColor', 'y');   % Make foreground patches red
%set(patches(26:50), 'FaceColor', 'w');  % Make background patches cyan
%set(findobj(get(gca, 'Children'), 'Type', 'Text'), 'FontSize', 12);
%set(patches(26:50), 'EdgeColor', 'k');
% figure
% subplot(n,n,1)
% for j=1:(n*n)
%     pos=(n*(n-1)+1)-(n*(idivide(int32(j-1),int32(n),'floor')))+mod((j-1),n);
%      clear k3 ka1 ka2 ka3
% k3=find(cl(:,:)==j);
%   subplot(n,n,double(pos)) 
%  STACKJ(j,:)=mean(X(:,k3),2);
%  hold on
%   plot(X(:,k3),'o-','LineWidth',0.1) 
%   plot(mean(X(:,k3),2),'k','LineWidth',2.0) 
%  title(j)
% end
