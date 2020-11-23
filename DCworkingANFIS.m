%% 
% By Piero Alessandro Riva Riquelme. 
% Undergraduate student. 
% Universidad de Concepción, Chile.
% pieroriva@udec.cl.
% November of 2020

%% Step 0: Cleaning
clear all
close all
clc
fprintf('ANFIS simulation started...');
tic

%% Step 1: Process definition
  fprintf('\nStep 1: Process definition...');
% a) Time definition
  fprintf('\n\ta) Time definition');
    ti = 0.01;
    step = 0.01;
    tf = 12;
    tt = ti:step:tf;
    nData = length(tt);
  
% b) Process definition (DC motor)
  fprintf('\n\tb) Process definition...');
  a1 = -0.680758164075105;
  a2 = -0.010439248103841;
  b0 = 9.217786558421309;
  theta = [a1 a2 b0];

% c) Generating input
  fprintf('\n\tc) Generating input...');
  ut = zeros(nData,1);
  h = @(t) heaviside(t);
  r = @(t) heaviside(t).*(t);
  uSelect = 'NoisyRamp';
  
  switch uSelect
      case 'NoisyRamp'
          u = @(t) -r(t)*4+r(t-3)*4+r(t-3)*4-r(t-9)*4-r(t-9)*4+r(t-12)*4;
          ut = u(tt);
          for i=1:nData
            ut(i) = ut(i)+(rand-0.5)*1.2;
          end
  end
  
% d) Generating output
 fprintf('\n\td) Generating output...');
 yt = zeros(nData,1);
 for i=3:nData
    yt(i) = -theta(1).*yt(i-1)-theta(2).*yt(i-2)+theta(3).*ut(i-1);
 end
 
 for i=2:nData
    yt(i) = yt(i)+0.2*(randn-0.5)*0; 
 end
 
 % Transform
 umin = -12;    % V
 umax = 12;     % V
 ymin = -360;   % rad/s
 ymax = 360;    % rad/s
 
 MIN = 10;      % % ? 4 bytes
 MAX = 90;      % % ? 4 bytes
 
 for i=1:nData
    ut(i) = (ut(i)-umin)/(2*umax);
    yt(i) = (yt(i)-ymin)/(2*ymax);
    
    ut(i) = ut(i)*(MAX-MIN)+MIN;
    yt(i) = yt(i)*(MAX-MIN)+MIN;
 end
 
% Deltas
 dut = zeros(nData,1);
 dyt = zeros(nData,1);
 for i=2:nData
     dut(i) = (ut(i)-ut(i-1))+50.0;
     dyt(i) = (yt(i)-yt(i-1))+50.0;     
 end
 
 % f) Plotting
 fprintf('\n\tf) Plotting...');
 figure(1);
 subplot(1,1,1); 
 cla
 hold on;
 plot(tt, ut, '-', 'LineWidth', 1);
 plot(tt, yt, '-', 'LineWidth', 1);
 title('Proccess output given input'); xlabel('Time (s)'); ylabel('(%)')
 grid minor; 
 legend('ut','yt');
 
 figure(2);
 subplot(1,1,1); 
 cla
 hold on;
 plot(tt, dut, '-', 'LineWidth', 1);
 plot(tt, dyt, '-', 'LineWidth', 1);
 title('Proccess output given input'); xlabel('Time (s)'); ylabel('(%)')
 grid minor; 
 legend('dut','dyt');
 
 

%% Step 2: Training configuration and space generation
  fprintf('\nStep 2: Training configuration and space generation...');
% a) Training configuration
  fprintf('\n\ta) Training configuration...');
  xx = 0:0.01:100;              % For plotting MFs
  nTrainingData = nData;        
  maxEpochs = 1000;            	% The more the merrier
  nInputs = 2;                  % Number of inputs, needs to be configured an ANFIS
  nFuzzy = 5;                   % Number of MFs per input
  nOutputs = 1;                 % Not changeable
  nRules = nFuzzy^nInputs;      % Number of rules
  K = 0.02;                     % Initial K
  maxK = 0.25;                 	% Maximum K, doesn't pass this number
  growthRate = 0.1;             % Growth of K 
  B = 15;                       % Backward values
  aIno = 20;                   	% Initial a parameter of premise MFs
  aOuto = 0.01;                 % Initial a parameter of consequent MFs
  tol = 1e-6;                 	% Tolerance for divisions by 0
  
% b) I/O
  OUTPUT = yt;      % Target function
  INPUT1 = yt;      % First input        
  INPUT2 = ut;      % Second input
  INPUT3 = yt;      % Third input
  INPUT4 = yt;      % Fourth input
  INPUT5 = ut;      % Fifth input
  INPUT6 = ut;      % Sixth input
  
% c) Tools
  fprintf('\n\tb) Generating tools...');
  gauss = @(x,a,c) exp(-(x-c).^2./(a.^2));      % Gauss MF for premise part
  sigmoid = @(x,a,c) 1./(1+exp(-(x-c)./a));     % Sigmoid MF for consequent par

  invSigmoid = @(x,a,c) c-a.*log(1./x-1);       % Inverse of sigmoid function

  dGauss_da = @(x,a,c) (2.*exp(-(-c+x).*(-c+x)./(a.*a)).*(-c+x).*(-c+x))./(a.*a.*a);    % Partial w/r to a
  dGauss_dc = @(x,a,c) (2.*exp(-(-c+x).*(-c+x)./(a.*a)).*(-c+x))./(a.*a);               % Partial w/r to c

  dinvSigmoid_da = @(x,a,c) -log(1./x-1);   % Partial of invSigmoid w/r to a
  dinvSigmoid_dc = @(x,a,c) 1;              % Partial of invSigmoid w/r to c

% d) Generating workspace
 fprintf('\n\tc) Generating workspace...');
  % d.1) Initial fuzzy parameters
    aIne  = zeros(nFuzzy,nInputs);
    cIne  = zeros(nFuzzy,nInputs);
    aOute = zeros(nRules,nOutputs);
    cOute = zeros(nRules,nOutputs);
    aInfinal  = aIne;
    cInfinal  = cIne;
    aOutfinal = aOute;
    cOutfinal = cOute;
    for i=1:nInputs
        aIne(:,i) = aIno.*ones(nFuzzy,1);   % Initial a for premise MFs
        cIne(:,i) = (0:(100/(nFuzzy-1)):100); % Initial c for premise MFs
    end
    for i=1:nOutputs
        aOute(:,i) = aOuto*ones(nRules,1);   % Initial a for consequent MFs
        cOute(:,i) = (0:(100/(nRules-1)):100); % Initial c for consequent MFs
    end

  % d.2) Training workspace
    APE  = zeros(maxEpochs,1);
    APEmin = 100000;
    epochFlag = 1;
    etaIna       = zeros(nFuzzy,nInputs);
    etaInc       = zeros(nFuzzy,nInputs);
    etaOuta      = zeros(nRules,1);
    etaOutc      = zeros(nRules,1);
    X           = zeros(nInputs,1);
    O5          = zeros(nTrainingData,1);
    En          = zeros(nTrainingData,1);
    muIn        = zeros(nFuzzy,nInputs);
    w           = zeros(nRules,1);
    wn          = zeros(nRules,1);
    fi          = zeros(nRules,1);
    fi_wn       = zeros(nRules,1);

    dJn_dO5     = zeros(nTrainingData,1);
    dJn_dO2     = zeros(nRules,1);
    dO5_dfi     = zeros(nRules,1);
    dO5_dO2     = zeros(1,nRules);
    dO2_dO1     = zeros(nRules,nFuzzy*nInputs);

    dfi_da      = zeros(nRules,1);
    dfi_dc      = zeros(nRules,1);
    dmu_daIn    = zeros(nFuzzy,nInputs);
    dmu_dcIn    = zeros(nFuzzy,nInputs);
    dJn_daOut     = zeros(nRules,1);
    dJn_dcOut     = zeros(nRules,1);
    dJp_daOut      = zeros(nRules,1);
    dJp_dcOut      = zeros(nRules,1);
    dJn_dmu     = zeros(nFuzzy,nInputs);
    dJn_daIn    = zeros(nFuzzy,nInputs);
    dJn_dcIn    = zeros(nFuzzy,nInputs);
    dJp_daIn     = zeros(nFuzzy,nInputs);
    dJp_dcIn     = zeros(nFuzzy,nInputs);
    
  % d.3) Index table
    indexTable  = zeros(nRules,nInputs); 
    for k = 1:nInputs
        l=1;
        for j=1:nFuzzy^(k-1):nRules
            for i =1:nFuzzy^(k-1)
                indexTable(j+i-1,nInputs-k+1)=l;
            end
            l=l+1;
            if l>nFuzzy
                l=1;
            end
        end
    end 

% e) Initial fuzzy sets plotting    
figure(3)
  for i=1:nInputs
     subplot(nOutputs+1,nInputs,i); 
     cla
     hold on
     for j=1:nFuzzy
        plot(xx,gauss(xx,aIne(j,i),cIne(j,i)),'DisplayName',[num2str(char(64+i)) num2str(j)],'LineWidth',0.5);
     end
     %legend;
     title(['Initial premise fuzzy set ' num2str(char(64+i))]); ylabel(['\mu(x_',num2str(i),')']); xlabel(['x_',num2str(i)]);
     grid minor;
  end
  
  for i=1:nOutputs
    subplot(nOutputs+1,nInputs,[(i*nInputs+1) (i*nInputs+nInputs)]);
    cla
    hold on;
    for j=1:nRules
        plot(xx,sigmoid(xx,aOute(j,i),cOute(j,i)),'LineWidth',0.1);
    end
    %legend;
    title(['Initial consequent fuzzy set Z_' num2str(i)]); ylabel(['\mu(z_',num2str(i),')']); xlabel(['z_',num2str(i)]);
    grid minor
  end 
    
    
%% Step 3: ANFIS training
fprintf('\nStep 3: ANFIS training begin...');
for g=1:maxEpochs
  dJp_daOut	= zeros(nRules,1);
  dJp_dcOut	= zeros(nRules,1);
  dJp_daIn  = zeros(nFuzzy,nInputs);
  dJp_dcIn	= zeros(nFuzzy,nInputs); 
	for i=1+B:nTrainingData
	  %% a) ANFIS;
	  % Prelayer: Selecting input variables
	    X(1)  = INPUT1(i-B);
        X(2)  = INPUT2(i);
        
	  % Layer 1: Input fuzzyfication  
        for k=1:nInputs
            for j=1:nFuzzy
                muIn(j,k) = gauss(X(k),aIne(j,k),cIne(j,k));
                if muIn(j,k)<tol
                   muIn(j,k) = tol;
                end
                if muIn(j,k)>(1.0-tol)
                   muIn(j,k) = 1.0-tol;
                end
            end            
        end

	  %Layer 2: Calculating weigths
        for j=1:nRules
                w(j) = 1.0;
            for k=1:nInputs
                %w(j) = min(w(j),muIn(indexTable(j,k),k)); % Not recommendable
                w(j) = w(j)*muIn(indexTable(j,k),k);
            end
        end
        
	  % Layer 3: Normalizing
        sumW = sum(w);
        if abs(sumW)<tol
            sumW=tol;
        end
        for j=1:nRules
           wn(j) = w(j)/sumW;
        end
       
	  % Layer 4: Calculating wn_i*f_i     
        for j=1:nRules
           fi(j)   = invSigmoid(w(j),aOute(j,1),cOute(j,1));
           fi_wn(j) = fi(j)*wn(j); 
        end

	  % Layer 5: Calculating output
        f = sum(fi_wn);
        O5(i) = f;

	  %% b) ANFIS error measure
        En(i) = OUTPUT(i)-O5(i);      % Measured error for the i-th data pair
	  % dJn/dO5
	    dJn_dO5(i) = -2*En(i);
         
      %% c) Gradient descent for consequent parameters
        for j=1:nRules
        %dO5_dfi
          dO5_dfi(j) = wn(j);
        %dfi_dalpha
          dfi_da(j) = dinvSigmoid_da(w(j),aOute(j,1),cOute(j,1));
          dfi_dc(j) = 1; %dinvSigmoid_dc(w(j),aOute(j,1),cOute(j,1));
        % dJn_dalpha
          dJn_daOut(j) = dJn_dO5(i)*dO5_dfi(j)*dfi_da(j);
          dJn_dcOut(j) = dJn_dO5(i)*dO5_dfi(j)*dfi_dc(j);
        % Sum
          dJp_daOut(j) = dJp_daOut(j)+dJn_daOut(j);
          dJp_dcOut(j) = dJp_dcOut(j)+dJn_dcOut(j);
        end     
        

      %% d) Gradient descent for premise parameters
	  % dO5/dO2       
        for j=1:nRules
          dO5_dO2(j) = (fi(j)-O5(i))/sumW; 
        end
	  % dO2_dO1 matrix
        for e=1:nInputs
            for k=1:nFuzzy
                for j=1:nRules
                    if(k==indexTable(j,e))
                        dO2_dO1(j,(e-1)*nFuzzy+k)=1.0;
                        for p=1:nInputs
                            if(e~=p)
                                %dO2_dO1(j,(e-1)*nFuzzy+k) = min(dO2_dO1(j,(e-1)*nFuzzy+k),muIn(indexTable(j,p),p)); % Not recommendable
                                dO2_dO1(j,(e-1)*nFuzzy+k) = dO2_dO1(j,(e-1)*nFuzzy+k)*muIn(indexTable(j,p),p);
                            end
                        end
                    else
                        dO2_dO1(j,(e-1)*nFuzzy+k)=0.0;
                    end
                end
            end
        end     
	  % dJn_dO2
      for j=1:nRules
        dJn_dO2(j) = dJn_dO5(i)*dO5_dO2(j);  
      end
        
	  % Chain rule
        for k=1:nInputs
          for j=1:nFuzzy
          % dJn_dO1
            dJn_dmu(j,k) = 0.0;
            for p=1:nRules
                dJn_dmu(j,k) = dJn_dmu(j,k)+ dJn_dO2(p)*dO2_dO1(p,j+(k-1)*nFuzzy);
            end
          % dO1_dalpha
            dmu_daIn(j,k)= dGauss_da(X(k),aIne(j,k),cIne(j,k));
            dmu_dcIn(j,k)= dGauss_dc(X(k),aIne(j,k),cIne(j,k));
          % dJn_dalpha
            dJn_daIn(j,k) = (dJn_dmu(j,k)).*dmu_daIn(j,k);
            dJn_dcIn(j,k) = (dJn_dmu(j,k)).*dmu_dcIn(j,k);
          % Sum
            dJp_daIn(j,k) = dJp_daIn(j,k)+dJn_daIn(j,k);
            dJp_dcIn(j,k) = dJp_dcIn(j,k)+dJn_dcIn(j,k);
          end
        end
    end
    
    %% e) Epoch summary
      APE(g,1) = 0;
      for i=1+B:nTrainingData
      if abs(OUTPUT(i))<1
          OUTPUT(i) = 1;
      end
          APE(g,1) = APE(g,1)+abs(En(i))/abs(OUTPUT(i));
      end
      APE(g,1) = APE(g,1)*100/(nTrainingData-B);    
      if APE(g,1)<=APEmin
        APEmin = APE(g,1);
        aInfinal = aIne;
        cInfinal = cIne;
        aOutfinal = aOute;
        cOutfinal = cOute;
        epochFlag = g;
      end
    
    %% f) New step size    
    if g>4
        if APE(g,1)<APE(g-1,1)
            if APE(g-1,1)<APE(g-2,1)
                if APE(g-2,1)<APE(g-3,1)
                    if APE(g-3,1)<APE(g-4,1)
                        K = K*(1.0+growthRate);
                    end
                end
            end
        else
            if APE(g-1,1)<APE(g-2,1)
                if APE(g-2,1)>APE(g-3,1)
                    if APE(g-3,1)<APE(g-4,1)
                    K=K*(1.0-growthRate);          
                    end
                end
            end
        end
    end
    if K>maxK
        K = maxK;
    end
    
    %% g) New consequent parameters
	for j=1:nRules
	  aOute(j,1) = aOute(j,1)-K*sign(dJp_daOut(j,1));
	  cOute(j,1) = cOute(j,1)-K*sign(dJp_dcOut(j,1));
      if abs(aOute(j,1))<tol
          aOute(j,1)=tol;
      end
      if abs(cOute(j,1))<tol
          cOute(j,1)=tol;
      end
    end
        
  %% h) New premise parameters
    for k=1:nInputs
      for j=1:nFuzzy
        aIne(j,k) = aIne(j,k)-K*sign(dJp_daIn(j,k));
        cIne(j,k) = cIne(j,k)-K*sign(dJp_dcIn(j,k));
        if abs(aIne(j,k))<tol
          aIne(j,k)=tol;
        end
        if abs(cIne(j,k))<tol
            cIne(j,k)=tol;
        end
      end
    end
    
  % fprintf  
   fprintf('\n\tEpoch: %04d.\t APE: %.10f.\t K: %.10f.\t APEmin: %d,\t %.7f',g,APE(g),K, epochFlag,APEmin);
  end



%% Step 4: Plotting training results
  fprintf('\n\tg): Plotting...');
  
  % a) APE evolution
  fprintf('\n\ta) APE evolution...');
  figure(4)
  hold on;
  subplot(2,1,1); 
  cla;
  hold on
  plot(1:length(APE(:,1)), APEmin.*ones(length(APE(:,1)),1), '--k', 'LineWidth', 0.1);
  plot(1:length(APE(:,1)), APE(:,1), '-', 'LineWidth', 1);
  ylim([0 max(APE)]);
  title('Average percentage error (APE) evolution'); xlabel('Epochs'); ylabel('APE (%)');
  legend('min','APE')
  grid minor;
  
  % b) Target and actual
  fprintf('\n\tb) Target and actual...');
  subplot(2,1,2);
  cla;
  hold on;
  legend('off');
  plot(tt, O5, '-b', 'LineWidth', 1.2); 
  plot(tt, OUTPUT, '-g', 'LineWidth', 1.2);   
  legend('Calculated','Target');
  title('Anfis training results'); xlabel('Time (s)'); ylabel('(%)');
  grid minor;
  
  % c) Best fuzzy sets
  fprintf('\n\tc) Best fuzzy sets...');
  figure(5)
  for i=1:nInputs
     subplot(nOutputs+1,nInputs,i); 
     hold on
     for j=1:nFuzzy
        plot(xx,gauss(xx,aInfinal(j,i),cInfinal(j,i)),'DisplayName',[num2str(char(64+i)) num2str(j)],'LineWidth',0.5);
     end
     %legend;
     title(['Best premise fuzzy set ' num2str(char(64+i))]); ylabel(['\mu(x_',num2str(i),')']); xlabel(['x_',num2str(i)]);
     grid minor;
  end
  
  for i=1:nOutputs
    subplot(nOutputs+1,nInputs,[(i*nInputs+1) (i*nInputs+nInputs)]);
    hold on;
    for j=1:nRules
        plot(xx,sigmoid(xx,aOutfinal(j,i),cOutfinal(j,i)),'LineWidth',0.1);
    end
    %legend;
    title(['Best consequent fuzzy set Z_' num2str(i)]); ylabel(['\mu(z_',num2str(i),')']); xlabel(['z_',num2str(i)]);
    grid minor
  end 

  % d) Training summary
  fprintf('\n\td) Training summary...');
  figure(6);
  cla;
  hold on;
  plot(tt,INPUT1(1:end), '--', 'LineWidth', 0.1,'Color',[0.4,0.4,0.4]); 
  plot(tt,INPUT2(1:end), '--', 'LineWidth', 0.1,'Color',[0.4,0.4,0.4]);  
  plot(tt,O5, '-b', 'LineWidth', 1.1); 
  plot(tt,OUTPUT, '-g', 'LineWidth', 1.1);
  title('ANFIS training summary'); xlabel('Time (s)'); ylabel('(%)');
  grid minor; 
  legend('ANFIS input1','ANFIS input2','ANFIS output','Target');

% e) Save results
  fprintf('\n\tb) Saving results...');
  save('workingANFIS.mat','aInfinal','cInfinal','aOutfinal','cOutfinal');
        
        
%% Step 5: ANFIS validation
  fprintf('\nStep5: ANFIS validation...');

% a) Time definition
  fprintf('\n\ta) Defining timeset...');
  tii =0.01;
  step =0.01;
  tff = 12;
  ttt= tii:step:tff;
  
% b) Input definition
  fprintf('\n\tb) Defining input...');
  uu = @(t) h(t)*50+h(t-0.5)*5-r(t-1)*16+r(t-3)*16+h(t-4)*5+h(t-5)*5+r(t-5.5)*40-r(t-6)*40+h(t-6.5)*5+...
            +h(t-7)*5+h(t-7.5)*5+h(t-8)*5+h(t-8.5)*5+h(t-9)*5;
  utt = zeros(length(ttt),1);
  utt(:,1) = uu(ttt);
  for i=1:length(utt)
     utt(i) = utt(i)+(rand-0.5)*0.2; 
  end

% Transform
  for i=1:length(utt)
     utt(i) = (utt(i)-MIN)/(MAX-MIN); 
     utt(i) = utt(i)*umax*2+umin; 
  end

% c) Process output definition
  fprintf('\n\tc) Calculating real process output...');
  ytt = zeros(length(utt),1);
  for i=3:length(utt)
     ytt(i) = -theta(1).*ytt(i-1)-theta(2).*ytt(i-2)+theta(3).*utt(i-1);
  end
 
 for i=2:length(utt)
    ytt(i) = ytt(i)+(randn-0.5)*0; 
 end

% Transform
for i=1:nData
    utt(i) = (utt(i)-umin)/(2*umax);
    ytt(i) = (ytt(i)-ymin)/(2*ymax);
    
    utt(i) = utt(i)*(MAX-MIN)+MIN;
    ytt(i) = ytt(i)*(MAX-MIN)+MIN;
 end


% d) ANFIS process
  fprintf('\n\td) Calculating ANFIS output...');
  yee = zeros(length(utt),1);
  for i=1:length(ytt)
        % Prelayer: Input variables
	    X(1)  = ytt(i);
        X(2)  = utt(i);
        % Layer 1: Fuzzification 
        for k=1:nInputs
            for j=1:nFuzzy
                muIn(j,k) = gauss(X(k),aInfinal(j,k),cInfinal(j,k));    % Better results
                if muIn(j,k)<tol
                   muIn(j,k) = tol;
                end
                if muIn(j,k)>1.0-tol
                   muIn(j,k) = 1.0-tol;
                end
            end            
        end
        %Layer 2: Calculation of w 
        for j=1:nRules
                w(j) = 1.0;
            for k=1:nInputs
                w(j) = w(j).*muIn(indexTable(j,k),k);
            end
        end
        % Layer 3: Normalization
        sumW = sum(w);
        if abs(sumW)<tol
            sumW=tol;
        end
        for j=1:nRules
           wn(j) = w(j)/sumW;
        end
        % Layer 4: Cálculo de wn_i*f_i 
        for j=1:nRules
           fi(j)   = invSigmoid(w(j),aOutfinal(j,1),cOutfinal(j,1)); % Better results
           fi_wn(j) = fi(j)*wn(j); 
        end
        % Layer 5: Final layer
         f = sum(fi_wn);
         yee(i) = f;
  end
 
% e) Correlation calculation
  correlation = corr(yee,ytt);
  fprintf('\n\tCorrelation is of: %.3f%%',correlation*100);
 
% f) Plotting
  fprintf('\n\tf) Plotting...\n');
  figure(7);
  cla;
  hold on;
  plot(ttt, ytt(1:end), '--', 'LineWidth', 0.1,'Color',[0.4,0.4,0.4]); 
  plot(ttt, utt(1:end), '--', 'LineWidth', 0.1,'Color',[0.4,0.4,0.4]);  

  plot(ttt, yee, '-b', 'LineWidth', 1); 
  plot(ttt, ytt, '-g', 'LineWidth', 1);
  title('ANFIS validation'); xlabel('Time (t)'); ylabel('y(t)');
  grid minor; 
  legend('ANFIS input1','ANFIS input2','ANFIS output','Target');  

%%              
fprintf('\nEND\n');
toc