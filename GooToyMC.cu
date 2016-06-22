#include "Variable.hh"
#include "FitManager.hh"
#include "UnbinnedDataSet.hh"
#include "BinnedDataSet.hh"
#include "ThreeBodiesPdf.hh"
#include "VoigtianPdf.hh"
#include "VoigtianThreshPdf.hh"
#include "FitControl.hh"
#include "TDatime.h"
#include "TH1F.h"
#include "TH2F.h"
#include "TMath.h"
#include "TStyle.h"
#include "TCanvas.h"
#include "TMinuit.hh"
#include "TPaveText.h"
#include "TString.h"
#include "AddPdf.hh"
#include "TRandom.hh"
#include "TRandom3.h"
#include "TLatex.h"
#include "TMultiGraph.h"
#include "TAxis.h"
#include "TColor.h"
#include "TFile.h"
#include "TGraph.h"
#include "TLegend.h"
#include "TPaveStats.h"

#include <assert.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include <sys/times.h>
#include <sys/time.h>
#include <fstream>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>

//#define SCATTERPLOTS 1
#define GOODPLOTS 30
//#define READING 1
#define NLLBEST 1
//#define NEGSIG 0.25
//#define GAMMAFIX 0.03
//#define MEANFIX 1051.4 
//#define TOYWRITE 1
//#define STARTINGPOINTS 1

using namespace std;

timeval startTime, stopTime, totalTime;
timeval startTimeRead, stopTimeRead, totalTimeRead;
clock_t startCPU, stopCPU;
clock_t startCPURead, stopCPURead;
tms startProc, stopProc;
tms startProcRead, stopProcRead;

////////////////////////////////////////////////////////////////////////////////////
  //BACKGROUND FUNCTION//

Double_t fondo (Double_t x) {

    Double_t ret = (x<1)||(x>2)? 0:(sqrt( pow(x+3.0967,4) + pow(3.0967,4) + pow(1.01946,4) - 2*pow(x+3.0967,2)*pow(3.0967,2) - 2*pow(3.0967,2)*pow(1.01946,2) - 2*pow(x+3.0967,2)*pow(1.01946,2) ) * sqrt( pow(5.279,4) + pow(x+3.0967,4) + pow(0.493677,4) - 2*pow(5.279,2)*pow(x+3.0967,2) - 2*pow(5.279,2)*pow(0.493677,2) - 2*pow(x+3.0967,2)*pow(0.493677,2) ) / (x+3.0967));

    return ret;
}

////////////////////////////////////////////////////////////////////////////////////
  //ROUND FUNCTION//

Double_t approximate (Double_t x) {

    Double_t result;
	
	if(x<=0) result = ((Double_t)floor(x*100000+0.5))/100000.0;
	 else return x;
	 
    return result;
}

////////////////////////////////////////////////////////////////////////////////////
  //CHISQUARE CALCULATOR//

Double_t chisquare(TH1F& dataHist,TH1F& pdfHist,Variable *xvar){

Double_t chi=0;

//#pragma omp parallel for
for(int y=0;y<=xvar->numbins;y++){

    Double_t data;
    Double_t pdf;
    Double_t termine=0;

   // pdf=floor(pdfHist.GetBinContent(y+1)+0.5); //ROUND TO INTEGER
   // data=floor(dataHist.GetBinContent(y+1)+0.5);
    pdf=pdfHist.GetBinContent(y+1); //ROUND TO INTEGER
    data=dataHist.GetBinContent(y+1);    
	
    termine=pow(pdf-data,2);

    if(pdf<1){
        chi+=termine;
    }else{
    termine/=pdf;
    chi+=termine;
    }
    //cout<<"Termine="<<termine<<" Data ="<<dataHist.GetBinContent(y+1)<<" PDF = "<<pdf<<" CHI = "<<chi<<endl;
}

return chi;

}

////////////////////////////////////////////////////////////////////////////////////

int main(int argc, char** argv) {
    //cudaSetDevice(0);
    cudaFree(0);
    CUresult a;
    CUcontext pctx;
    CUdevice device;
/*    cuInit(0);
    cuDeviceGet(&device, 0);
    std::cout << "DeviceGet : " << a << std::endl;
    //cuCtxCreate(&pctx, CU_CTX_SCHED_AUTO, device ); // explicit context here
    std::cout << "CtxCreate : " << a << std::endl;
    assert(a == CUDA_SUCCESS);
    a = cuCtxPopCurrent(&pctx);
    std::cout << "cuCtxPopCurrent : " << a << std::endl;
    //cuCtxDetach(pctx);
    //cuCtxDetach(pctx);
    //a = cuCtxPopCurrent(&pctx);
    //assert(a == CUDA_SUCCESS);
    std::cout << "Initialized CUDA" << std::endl;
 
   //cuInit();
    cudaDeviceReset();
   // cudaSetDevice(0);
    gStyle->SetOptStat(1111111111);
  */  gStyle->SetFillColor(0);
////////////////////////////////////////////////////////////////////////////////////
//INPUTS
////////////////////////////////////////////////////////////////////////////////////
  //Events & toys
  int events=2500;//2543;//5000;//
  int iter = atoi(argv[1]);
  
  //Randomizer MultiProcesses
  int rndInt = 1;
  #ifndef READING
  if(argc>2) rndInt = atoi(argv[2]);
  else rndInt = 5;
  #endif
  
  //Reader MultiProcesses
  int toys = 0;
  #ifdef READING
  if(argc>2) toys = atoi(argv[2]); //Dividing Toys Among MultiProcesses  
  #endif
  #ifndef READING
  if(argc>3) toys = atoi(argv[3]); 
  #endif
////////////////////////////////////////////////////////////////////////////////////

  TString name = "";
  switch (iter) {
  case 100: name = "100";
    break;
  case 1000: name = "1k";
    break;
  case 10000: name = "10k";
    break;
  case 100000: name = "100k";
    break;
  case 500000: name = "500k";
    break;
  case 1000000: name = "1M";
    break;
  case 5000000: name = "5M";
    break;
  case 10000000: name = "10M";
    break;
  case 50000000: name = "50M";
    break;
  case 100000000: name = "100M";
    break;

  default: name = argv[1];
    break;
  }


  cout<<endl;
  cout<<"==================================================================================================="<<endl;
  cout<<"=========================== Starting Toy MC : "<<name<<" iterations ==========================="<<endl;
  cout<<"==================================================================================================="<<endl;
  cout<<endl;

////////////////////////////////////////////////////////////////////////////////////
  //MAIN VARIABLE//
  Variable* xvar = new Variable("Mass",1.02,0.001,1.008, 1.568);
  xvar->numbins = 28;

////////////////////////////////////////////////////////////////////////////////////
  //NAMES
  char canvasname[256];
  char bufferstring[256];
  char filename[256];
  char pdfname[256];
  char histoname[256];
  char dircname[256];
  int gFix=0; 
  int mFix=0; 
  int sNeg=0;
  
////////////////////////////////////////////////////////////////////////////////////
  //PHYSICAL PARAMETERS//
  float MassStart = 1.0467 - 0.003*3.0;
  float MassEnd = 1.0467 + 0.003*3.0;
  float myWidthStart =  0.005;//0.017; //0.01
  float myWidthEnd = 0.0605; //0.08
  float GammaStart = 0.01; //0.0017;
  float GammaEnd = 0.0653;

  #ifdef SCATTERPLOTS
////////////////////////////////////////////////////////////////////////////////////
   //GRID FOR SCATTERPLOTS//
////////////////////////////////////////////////////////////////////////////////////
   TMultiGraph *grid = new TMultiGraph();

////////////////////////////////////////////////////////////////////////////////////
 //STARTING POINTS GRAPHS//
////////////////////////////////////////////////////////////////////////////////////
    //Double_t Width1[1]={0.005};
	 Double_t Width1[1]={0.015};
    Double_t Width2[1]={0.0235};
    Double_t Width3[1]={0.0420};
    Double_t Width4[1]={0.0605};

    Double_t Massa1[1]={1.0407};
    Double_t Massa2[1]={1.0507};

    TGraph* Start1 = new TGraph(1,Massa1,Width1);
    Start1->SetMarkerStyle(29);
    Start1->SetMarkerColor(1);
    Start1->SetMarkerSize(1.3);

    TGraph* Start2 = new TGraph(1,Massa2,Width1);
    Start2->SetMarkerStyle(29);
    Start2->SetMarkerColor(2);
    Start2->SetMarkerSize(1.3);
   
    TGraph* Start3 = new TGraph(1,Massa1,Width2);
    Start3->SetMarkerStyle(29);
    Start3->SetMarkerColor(3);
    Start3->SetMarkerSize(1.3);

    TGraph* Start4 = new TGraph(1,Massa2,Width2);
    Start4->SetMarkerStyle(29);
    Start4->SetMarkerColor(4);
    Start4->SetMarkerSize(1.3);

    TGraph* Start5 = new TGraph(1,Massa1,Width3);
    Start5->SetMarkerStyle(29);
    Start5->SetMarkerColor(9);
    Start5->SetMarkerSize(1.3);
    
    TGraph* Start6 = new TGraph(1,Massa2,Width3);
    Start6->SetMarkerStyle(29);
    Start6->SetMarkerColor(6);
    Start6->SetMarkerSize(1.3);

    TGraph* Start7 = new TGraph(1,Massa1,Width4);
    Start7->SetMarkerStyle(29);
    Start7->SetMarkerColor(7);
    Start7->SetMarkerSize(1.3);
    
    TGraph* Start8 = new TGraph(1,Massa2,Width4);
    Start8->SetMarkerStyle(29);
    Start8->SetMarkerColor(8);
    Start8->SetMarkerSize(1.3);

////////////////////////////////////////////////////////////////////////////////////
//AXES FOR GRID (BINNING)
////////////////////////////////////////////////////////////////////////////////////

	 Double_t y1[2] = {0.02,0.02};
    Double_t y2[2] = {0.04,0.04};
    Double_t y3[2] = {0.06,0.06};

    Double_t x1[2]={1.048,1.048};

    Double_t MeanLim[2]={MassStart,MassEnd};
    Double_t MeanMax[2]={MassEnd,MassEnd};
    Double_t MeanMin[2]={MassStart,MassStart};
    Double_t GammaLim[2]={GammaStart,GammaEnd};
    Double_t GammaMin[2]={GammaStart,GammaStart};
    Double_t GammaMax[2]={GammaEnd,GammaEnd};

   //X-axises//
       TGraph* XGrid1 = new TGraph(2,MeanLim,y1);
       TGraph* XGrid2 = new TGraph(2,MeanLim,y2);
       TGraph* XGrid3 = new TGraph(2,MeanLim,y3);
       TGraph* XMin = new TGraph(2,MeanMin,GammaLim);
       TGraph* XMax = new TGraph(2,MeanMax,GammaLim);
	   
	    XGrid1->SetLineWidth(1);
       XGrid1->SetLineStyle(3);
       XGrid1->SetLineColor(31);

       XGrid3->SetLineWidth(1);
       XGrid3->SetLineStyle(3);
       XGrid3->SetLineColor(31);

       XGrid2->SetLineWidth(1);
       XGrid2->SetLineStyle(3);
       XGrid2->SetLineColor(31);

      
	   
	   XMin->SetLineWidth(1);
       XMin->SetLineStyle(3);
       XMin->SetLineColor(31);

       XMax->SetLineWidth(1);
       XMax->SetLineStyle(3);
       XMax->SetLineColor(31);

    //Y-axises//
       TGraph* YGrid1 = new TGraph(2,x1,GammaLim);
       TGraph* YMin = new TGraph(2,MeanLim,GammaMin);
       TGraph* YMax = new TGraph(2,MeanLim,GammaMax);
	    
		 YGrid1->SetLineWidth(1);
        YGrid1->SetLineStyle(3);
        YGrid1->SetLineColor(31);
	   
	    YMin->SetLineWidth(1);
       YMin->SetLineStyle(3);
       YMin->SetLineColor(31);
	   
	    YMax->SetLineWidth(1);
       YMax->SetLineStyle(3);
       YMax->SetLineColor(31);


////////////////////////////////////////////////////////////////////////////////////

       grid->Add(XGrid1,"L");
       grid->Add(XGrid2,"L");
       grid->Add(XGrid3,"L");
       grid->Add(YGrid1,"L");
       grid->Add(YMax,"L");
       grid->Add(YMin,"L");
       grid->Add(XMax,"L");
       grid->Add(XMin,"L");


       grid->Add(Start1,"P");
       grid->Add(Start2,"P");
       grid->Add(Start3,"P");
       grid->Add(Start4,"P");
       grid->Add(Start5,"P");
       grid->Add(Start6,"P");
       grid->Add(Start7,"P");
       grid->Add(Start8,"P");

#endif
////////////////////////////////////////////////////////////////////////////////////
   //TIME VARIABLES//
  TDatime *starttime = new TDatime();
  Int_t Date = starttime->GetDate();
  Int_t Clock = starttime->GetTime();
  gettimeofday(&startTime, NULL);
  startCPU = times(&startProc);

////////////////////////////////////////////////////////////////////////////////////
//RANDOMIZER FOR MULTIPLE PROCESS//
////////////////////////////////////////////////////////////////////////////////////

  timeval trand;
  gettimeofday(&trand,NULL);
  long int msRand = trand.tv_sec * 1000 + trand.tv_usec / 1000;
  TRandom3 fileran(msRand);
  Double_t randomize = fileran.Uniform(fileran.Uniform(fileran.Uniform(13.0)))+fileran.Uniform(2.0);
  rndInt *= (int)(100*randomize);

  //START TIME//
  gettimeofday(&startTime, NULL);
  startCPU = times(&startProc);

////////////////////////////////////////////////////////////////////////////////////
//FILES
////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////
//Fixed Parameters
  #ifdef GAMMAFIX
  gFix=1;
  #endif
  
  #ifdef MEANFIX
  mFix=1;
  #endif
  
  #ifdef NEGSIG
  sNeg=1;
  #endif

////////////////////////////////////////////////////////////////////////////////////  
  //CHIS TXT FILE
  //Organizing in Directories
  sprintf(dircname,"mkdir ./txt_files_imp/%d/",Date);
  system(dircname);
  sprintf(dircname,"mkdir ./txt_files_imp/%d/%d/",Date,(int)iter);
  system(dircname);
  sprintf(dircname,"mkdir ./txt_files_imp/%d/%d/%d%d%d/",Date,(int)iter,gFix,mFix,sNeg);
  system(dircname);
  
  #ifndef NLLBEST
  #ifdef READING
  sprintf(filename,"txt_files_imp/%d/%d/%d%d%d/%d-ToyMCDeltaChisGoox%d%d%d-%d-%d-%d.txt",Date,iter,gFix,mFix,sNeg,iter,gFix,mFix,sNeg,Date,Clock,toys);
  ofstream chiFile(filename);
  #else
  sprintf(filename,"txt_files_imp/%d-ToyMCDeltaChisGoox%d%d%d-%d-%d-%.2f.txt",Date,iter,gFix,mFix,sNeg,iter,gFix,mFix,sNeg,Date,Clock,randomize);
  ofstream chiFile(filename);
  #endif
  #endif
  
  
  #ifdef NLLBEST
  #ifdef READING
  sprintf(filename,"txt_files_imp/%d/%d/%d%d%d/%d-ToyMCDeltaChisGooNLLx%d%d%d-%d-%d-%d.txt",Date,iter,gFix,mFix,sNeg,iter,gFix,mFix,sNeg,Date,Clock,toys);
  ofstream chiFileNLL(filename);
  //sprintf(filename,"txt_files_imp/%d/%d/%d%d%d/%d-ToyMCNullNLLGoox%d%d%d-%d-%d-%d.txt",Date,iter,gFix,mFix,sNeg,iter,gFix,mFix,sNeg,Date,Clock,toys);
  //ofstream nullNLLFile(filename);
  sprintf(filename,"txt_files_imp/%d/%d/%d%d%d/%d-ToyMCNLLGoox%d%d%d-%d-%d-%d.txt",Date,iter,gFix,mFix,sNeg,iter,gFix,mFix,sNeg,Date,Clock,toys);
  ofstream fileNLL(filename);
  #else
  sprintf(filename,"txt_files_imp/%d/%d/%d%d%d/%d-ToyMCDeltaChisGooNLLx%d%d%d-%d-%d-%d-%.3f.txt",Date,iter,gFix,mFix,sNeg,iter,gFix,mFix,sNeg,Date,Clock,toys,randomize);
  ofstream chiFileNLL(filename);
  //sprintf(filename,"txt_files_imp/%d/%d/%d%d%d/%d-ToyMCNullNLLGoox%d%d%d-%d-%d-%d-%.3f.txt",Date,iter,gFix,mFix,sNeg,iter,gFix,mFix,sNeg,Date,Clock,toys,randomize);
  //ofstream nullNLLFile(filename);
  sprintf(filename,"txt_files_imp/%d/%d/%d%d%d/%d-ToyMCNLLGoox%d%d%d-%d-%d-%d-%.3f.txt",Date,iter,gFix,mFix,sNeg,iter,gFix,mFix,sNeg,Date,Clock,toys,randomize);
  ofstream fileNLL(filename);
  #endif
  #endif
  //sprintf(filename,"txt_files_imp/%d-ChiCompare-%d.txt",iter,Clock);
  //ofstream chiCompare(filename);
  
////////////////////////////////////////////////////////////////////////////////////  
  //ROOT FILE
  
  sprintf(dircname,"mkdir ./Histos/%d/",Date);
  system(dircname);
  sprintf(dircname,"mkdir ./Histos/%d/%d/",Date,(int)iter);
  system(dircname);
  #ifdef READING
  sprintf(filename,"./Histos/%d/%d/%dRunsToyMCx%d%d%d-%d-%d-%d.root",Date,iter,iter,gFix,mFix,sNeg,Date,Clock,toys);
  #else
  sprintf(filename,"./Histos/%d/%d/%dRunsToyMCx%d%d%d-%d-%d-%d-%f.root",Date,iter,iter,gFix,mFix,sNeg,Date,Clock,toys,randomize);
  #endif
  sprintf(bufferstring,"ToyMC %d Runs ",iter);
  TFile GooFile(filename,"RECREATE",bufferstring);

////////////////////////////////////////////////////////////////////////////////////  
  //INPUT FILE
  #ifdef READING
  sprintf(filename,"./Input/20150425/%d/%dToysGenerated%d.root",iter,iter,toys);
  //sprintf(filename,"./Input/%d/%d/ToysGenerated%d.root",Date,iter,toys);
  TFile* fileInput = TFile::Open(filename);
  #endif
  
////////////////////////////////////////////////////////////////////////////////////
  //TOYS FILE
  #ifdef TOYWRITE
  sprintf(dircname,"mkdir ./Input/%d/",Date);
  system(dircname);
  
  sprintf(dircname,"mkdir ./Input/%d/%d/",Date,(int)iter);
  system(dircname);
		
  sprintf(filename,"./Input/%d/%d/%dRunsToysGenerated-%d-%f.root",Date,iter,iter,Date,randomize);
  sprintf(bufferstring,"ToyMC %d Runs Datasets",iter);
  TFile ToyGenFile(filename,"RECREATE",bufferstring);
  #endif

////////////////////////////////////////////////////////////////////////////////////
  //CANVAS
  TCanvas* canvas = new TCanvas("","",1200,1000);

////////////////////////////////////////////////////////////////////////////////////
  //DATASETS//
  BinnedDataSet dataNull(xvar);
  BinnedDataSet dataSig(xvar);

////////////////////////////////////////////////////////////////////////////////////
  //PHYSICS
  #ifdef GAMMAFIX
  Variable *Gamma = new Variable("Gamma",GAMMAFIX);
  #endif
  #ifndef GAMMAFIX
  Variable *Gamma = new Variable("Gamma",0.0150,0.00001,GammaStart,GammaEnd);
  #endif
  
  #ifndef MEANFIX
  Variable *Mean = new Variable("Mean",1.040,0.0001,MassStart,MassEnd);
  #endif
  #ifdef MEANFIX
  Variable *Mean = new Variable("Mean",MEANFIX);
  #endif
  
  Variable* Sigma = new Variable("Sigma", 0.002);

////////////////////////////////////////////////////////////////////////////////////
  //HISTOGRAMS//
  #ifndef READING
  TH1F* genHist = new TH1F("genHist", "",xvar->numbins, xvar->lowerlimit, xvar->upperlimit); //DATA HISTO
  #endif
  #ifdef READING
  TH1F* genHist;
  #endif
  
  TH1F pdfNullHist("pdfNullHist", "", xvar->numbins, xvar->lowerlimit, xvar->upperlimit); //NULL PDF HISTO
  TH1F pdfSigHist("pdfSignalHist", "", xvar->numbins, xvar->lowerlimit, xvar->upperlimit);
  
  vector<TH1F> pdfSigHistos;
  TH1F pdfSig1Hist("pdfSig1Hist", "", xvar->numbins, xvar->lowerlimit, xvar->upperlimit);
  pdfSigHistos.push_back(pdfSig1Hist);
  TH1F pdfSig2Hist("pdfSig2Hist", "", xvar->numbins, xvar->lowerlimit, xvar->upperlimit);
  pdfSigHistos.push_back(pdfSig2Hist);
  TH1F pdfSig3Hist("pdfSig3Hist", "", xvar->numbins, xvar->lowerlimit, xvar->upperlimit);
  pdfSigHistos.push_back(pdfSig3Hist);
  TH1F pdfSig4Hist("pdfSig4Hist", "", xvar->numbins, xvar->lowerlimit, xvar->upperlimit);
  pdfSigHistos.push_back(pdfSig4Hist);
  TH1F pdfSig5Hist("pdfSig5Hist", "", xvar->numbins, xvar->lowerlimit, xvar->upperlimit);
  pdfSigHistos.push_back(pdfSig5Hist);
  TH1F pdfSig6Hist("pdfSig6Hist", "", xvar->numbins, xvar->lowerlimit, xvar->upperlimit);
  pdfSigHistos.push_back(pdfSig6Hist);
  TH1F pdfSig7Hist("pdfSig7Hist", "", xvar->numbins, xvar->lowerlimit, xvar->upperlimit);
  pdfSigHistos.push_back(pdfSig7Hist);
  TH1F pdfSig8Hist("pdfSig8Hist", "", xvar->numbins, xvar->lowerlimit, xvar->upperlimit);
  pdfSigHistos.push_back(pdfSig8Hist);
  TH1F* pdfSigBestHist = new TH1F("pdfSigBestHist", "", xvar->numbins, xvar->lowerlimit, xvar->upperlimit);
  
  #ifdef GOODPLOTS 
  //TH1F pdfNullHistPlot("pdfNullHistPlot", "",PLOTTINGFINENESS, xvar->lowerlimit, xvar->upperlimit); //NULL PDF HISTO
  //TH1F pdfSigHistPlot("pdfSigHistPlot", "",PLOTTINGFINENESS, xvar->lowerlimit, xvar->upperlimit);
  
	//BEST PARAMETERS TXT FILE
  sprintf(filename,"txt_files/%d-%d-ToyMCPlotParaMetersGoo-%d.txt",Date,Clock,iter);
  ofstream paramFile(filename);

  #endif
  
  #ifdef STARTINGPOINTS
  TH1I* startingPoints = new TH1I("Starting Points", "", 10,0,10);
  TH1I* startingPointsNLL = new TH1I("Starting Points Nll", "", 10,0,10);
  #endif
  
  #ifdef SCATTERPLOTS
  Int_t scatterBin = 100;
  Double_t gammaLow = GammaStart - GammaStart*0.5;
  Double_t gammaHigh = GammaEnd + GammaStart*0.5;
  Double_t meanLow = MassStart-MassStart*0.001;
  Double_t meanHigh = MassEnd+MassStart*0.001;
  
//////////////////////////////////////////////////////////////////////////////////////////////////////////
//TOTAL SCATTER PLOTS////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////
//CHI vs SIG
  
  TH2F scatterPlotChiSigBest("scatterSigBest","Chi vs Signal Fraction",100,0,0.05,100,0,30);
  TH2F scatterPlotChiSigBestSLim("scatterPlotChiSigBestSLim","Chi vs Signal Fraction",100,0,0.05,100,0,30);
//CHI vs MEAN
 
  TH2F scatterPlotChiMeanBest("scatterPlotChiMeanBest","Chi vs Mean ",100,1.035,1.065,100,0,30);
  TH2F scatterPlotChiMeanBestSLim("scatterPlotChiMeanBestSLim","Chi vs Mean ",100,1.035,1.065,100,0,30);
  TH2F scatterPlotChiMeanBestMLim("scatterPlotChiMeanBestMLim","Chi vs Mean ",100,1.035,1.065,100,0,30);
  TH2F scatterPlotChiMeanBestGLim("scatterPlotChiMeanBestGLim","Chi vs Mean ",100,1.035,1.065,100,0,30);
//CHI vs GAMMA
  
  TH2F scatterPlotChiGammaBest("scatterPlotChiGammaBest","Chi vs Gamma ",100,0.001,0.07,100,0,30);
  TH2F scatterPlotChiGammaBestSLim("scatterPlotChiGammaBestSLim","Chi vs Gamma ",100,0.001,0.07,100,0,30);
  TH2F scatterPlotChiGammaBestMLim("scatterPlotChiGammaBestMLim","Chi vs Gamma ",100,0.001,0.07,100,0,30);
  TH2F scatterPlotChiGammaBestGLim("scatterPlotChiGammaBestGLim","Chi vs Gamma ",100,0.001,0.07,100,0,30);
//SIG vs MEAN

  TH2F scatterPlotSigMeanBest("scatterPlotSigMeanBest","Signal Fraction vs Mean ",100,1.035,1.065,100,0,0.05);
  TH2F scatterPlotSigMeanBestSLim("scatterPlotSigMeanBestSLim","Signal Fraction vs Mean ",100,1.035,1.065,100,0,0.05);
  TH2F scatterPlotSigMeanBestMLim("scatterPlotSigMeanBestMLim","Signal Fraction vs Mean ",100,1.035,1.065,100,0,0.05);
//SIG vs GAMMA

  TH2F scatterPlotSigGammaBest("scatterPlotSigGammaBest","Signal Fraction vs Gamma",100,0.001,0.07,100,0,0.05);
  TH2F scatterPlotSigGammaBestSLim("scatterPlotSigGammaBestSLim","Signal Fraction vs Gamma",100,0.001,0.07,100,0,0.05);
  TH2F scatterPlotSigGammaBestGLim("scatterPlotSigGammaBestGLim","Signal Fraction vs Gamma",100,0.001,0.07,100,0,0.05);
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//STARTING POINTS SCATTER PLOTS GAMMA -MEAN ///////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  
  vector<TH1F> deltaChiStarts;
  TH1F ToyDelta1("ToyDelta1","Toy MC Delta Chi Square 1",5,0,0.0001);
  deltaChiStarts.push_back(ToyDelta1);
  TH1F ToyDelta2("ToyDelta2","Toy MC Delta Chi Square 2",5,0,0.0001);
  deltaChiStarts.push_back(ToyDelta2);
  TH1F ToyDelta3("ToyDelta3","Toy MC Delta Chi Square 3",5,0,0.0001);
  deltaChiStarts.push_back(ToyDelta3);
  TH1F ToyDelta4("ToyDelta4","Toy MC Delta Chi Square 4",5,0,0.0001);
  deltaChiStarts.push_back(ToyDelta4);
  TH1F ToyDelta5("ToyDelta5","Toy MC Delta Chi Square 5",5,0,0.0001);
  deltaChiStarts.push_back(ToyDelta5);
  TH1F ToyDelta6("ToyDelta6","Toy MC Delta Chi Square 6",5,0,0.0001);
  deltaChiStarts.push_back(ToyDelta6);
  TH1F ToyDelta7("ToyDelta7","Toy MC Delta Chi Square 7",5,0,0.0001);
  deltaChiStarts.push_back(ToyDelta7);
  TH1F ToyDelta8("ToyDelta8","Toy MC Delta Chi Square 8",5,0,0.0001);
  deltaChiStarts.push_back(ToyDelta8);
  
  vector<TH2F> scatterPlotGammaMean;
  TH2F scatterPlotGammaMean1("scatterPlotGammaMean1","Gamma vs Mean Start 1",scatterBin,meanLow,meanHigh,scatterBin,gammaLow,gammaHigh);
  scatterPlotGammaMean.push_back(scatterPlotGammaMean1);
  TH2F scatterPlotGammaMean2("scatterPlotGammaMean2","Gamma vs Mean Start 2",scatterBin,meanLow,meanHigh,scatterBin,gammaLow,gammaHigh);
  scatterPlotGammaMean.push_back(scatterPlotGammaMean2);
  TH2F scatterPlotGammaMean3("scatterPlotGammaMean3","Gamma vs Mean Start 3",scatterBin,meanLow,meanHigh,scatterBin,gammaLow,gammaHigh);
  scatterPlotGammaMean.push_back(scatterPlotGammaMean3);
  TH2F scatterPlotGammaMean4("scatterPlotGammaMean4","Gamma vs Mean Start 4",scatterBin,meanLow,meanHigh,scatterBin,gammaLow,gammaHigh);
  scatterPlotGammaMean.push_back(scatterPlotGammaMean4);
  TH2F scatterPlotGammaMean5("scatterPlotGammaMean5","Gamma vs Mean Start 5",scatterBin,meanLow,meanHigh,scatterBin,gammaLow,gammaHigh);
  scatterPlotGammaMean.push_back(scatterPlotGammaMean5);
  TH2F scatterPlotGammaMean6("scatterPlotGammaMean6","Gamma vs Mean Start 6",scatterBin,meanLow,meanHigh,scatterBin,gammaLow,gammaHigh);
  scatterPlotGammaMean.push_back(scatterPlotGammaMean6);
  TH2F scatterPlotGammaMean7("scatterPlotGammaMean7","Gamma vs Mean Start 7",scatterBin,meanLow,meanHigh,scatterBin,gammaLow,gammaHigh);
  scatterPlotGammaMean.push_back(scatterPlotGammaMean7);
  TH2F scatterPlotGammaMean8("scatterPlotGammaMean8","Gamma vs Mean Start 8",scatterBin,meanLow,meanHigh,scatterBin,gammaLow,gammaHigh);
  scatterPlotGammaMean.push_back(scatterPlotGammaMean8);
  
  vector<TH2F> scatterPlotGammaMeanSLim;
  TH2F scatterPlotGammaMean1SLim("scatterPlotGammaMean1SLim","Gamma vs Mean 1 SLim",scatterBin,meanLow,meanHigh,scatterBin,gammaLow,gammaHigh);
  scatterPlotGammaMeanSLim.push_back(scatterPlotGammaMean1SLim);
  TH2F scatterPlotGammaMean2SLim("scatterPlotGammaMean2SLim","Gamma vs Mean 2 SLim",scatterBin,meanLow,meanHigh,scatterBin,gammaLow,gammaHigh);
  scatterPlotGammaMeanSLim.push_back(scatterPlotGammaMean2SLim);
  TH2F scatterPlotGammaMean3SLim("scatterPlotGammaMean3SLim","Gamma vs Mean 3 SLim",scatterBin,meanLow,meanHigh,scatterBin,gammaLow,gammaHigh);
  scatterPlotGammaMeanSLim.push_back(scatterPlotGammaMean3SLim);
  TH2F scatterPlotGammaMean4SLim("scatterPlotGammaMean4SLim","Gamma vs Mean 4 SLim",scatterBin,meanLow,meanHigh,scatterBin,gammaLow,gammaHigh);
  scatterPlotGammaMeanSLim.push_back(scatterPlotGammaMean4SLim);
  TH2F scatterPlotGammaMean5SLim("scatterPlotGammaMean5SLim","Gamma vs Mean 5 SLim",scatterBin,meanLow,meanHigh,scatterBin,gammaLow,gammaHigh);
  scatterPlotGammaMeanSLim.push_back(scatterPlotGammaMean5SLim);
  TH2F scatterPlotGammaMean6SLim("scatterPlotGammaMean6SLim","Gamma vs Mean 6 SLim",scatterBin,meanLow,meanHigh,scatterBin,gammaLow,gammaHigh);
  scatterPlotGammaMeanSLim.push_back(scatterPlotGammaMean6SLim);
  TH2F scatterPlotGammaMean7SLim("scatterPlotGammaMean7SLim","Gamma vs Mean 7 SLim",scatterBin,meanLow,meanHigh,scatterBin,gammaLow,gammaHigh);
  scatterPlotGammaMeanSLim.push_back(scatterPlotGammaMean7SLim);
  TH2F scatterPlotGammaMean8SLim("scatterPlotGammaMean8SLim","Gamma vs Mean 8 SLim",scatterBin,meanLow,meanHigh,scatterBin,gammaLow,gammaHigh);
  scatterPlotGammaMeanSLim.push_back(scatterPlotGammaMean8SLim);
   
   
  vector<TH2F> scatterPlotGammaMeanMLim;
  TH2F scatterPlotGammaMean1MLim("scatterPlotGammaMean1MLim","Gamma vs Mean 1 MLim",scatterBin,meanLow,meanHigh,scatterBin,gammaLow,gammaHigh);
  scatterPlotGammaMeanMLim.push_back(scatterPlotGammaMean1MLim);
  TH2F scatterPlotGammaMean2MLim("scatterPlotGammaMean2MLim","Gamma vs Mean 2 MLim",scatterBin,meanLow,meanHigh,scatterBin,gammaLow,gammaHigh);
  scatterPlotGammaMeanMLim.push_back(scatterPlotGammaMean2MLim);
  TH2F scatterPlotGammaMean3MLim("scatterPlotGammaMean3MLim","Gamma vs Mean 3 MLim",scatterBin,meanLow,meanHigh,scatterBin,gammaLow,gammaHigh);
  scatterPlotGammaMeanMLim.push_back(scatterPlotGammaMean3MLim);
  TH2F scatterPlotGammaMean4MLim("scatterPlotGammaMean4MLim","Gamma vs Mean 4 MLim",scatterBin,meanLow,meanHigh,scatterBin,gammaLow,gammaHigh);
  scatterPlotGammaMeanMLim.push_back(scatterPlotGammaMean4MLim);
  TH2F scatterPlotGammaMean5MLim("scatterPlotGammaMean5MLim","Gamma vs Mean 5 MLim",scatterBin,meanLow,meanHigh,scatterBin,gammaLow,gammaHigh);
  scatterPlotGammaMeanMLim.push_back(scatterPlotGammaMean5MLim);
  TH2F scatterPlotGammaMean6MLim("scatterPlotGammaMean6MLim","Gamma vs Mean 6 MLim",scatterBin,meanLow,meanHigh,scatterBin,gammaLow,gammaHigh);
  scatterPlotGammaMeanMLim.push_back(scatterPlotGammaMean6MLim);
  TH2F scatterPlotGammaMean7MLim("scatterPlotGammaMean7MLim","Gamma vs Mean 7 MLim",scatterBin,meanLow,meanHigh,scatterBin,gammaLow,gammaHigh);
  scatterPlotGammaMeanMLim.push_back(scatterPlotGammaMean7MLim);
  TH2F scatterPlotGammaMean8MLim("scatterPlotGammaMean8MLim","Gamma vs Mean 8 MLim",scatterBin,meanLow,meanHigh,scatterBin,gammaLow,gammaHigh);
  scatterPlotGammaMeanMLim.push_back(scatterPlotGammaMean8MLim);
  
  vector<TH2F> scatterPlotGammaMeanGLim;
  TH2F scatterPlotGammaMean1GLim("scatterPlotGammaMean1GLim","Gamma vs Mean 1 GLim",scatterBin,meanLow,meanHigh,scatterBin,gammaLow,gammaHigh);
  scatterPlotGammaMeanGLim.push_back(scatterPlotGammaMean1GLim);
  TH2F scatterPlotGammaMean2GLim("scatterPlotGammaMean2GLim","Gamma vs Mean 2 GLim",scatterBin,meanLow,meanHigh,scatterBin,gammaLow,gammaHigh);
  scatterPlotGammaMeanGLim.push_back(scatterPlotGammaMean2GLim);
  TH2F scatterPlotGammaMean3GLim("scatterPlotGammaMean3GLim","Gamma vs Mean 3 GLim",scatterBin,meanLow,meanHigh,scatterBin,gammaLow,gammaHigh);
  scatterPlotGammaMeanGLim.push_back(scatterPlotGammaMean3GLim);
  TH2F scatterPlotGammaMean4GLim("scatterPlotGammaMean4GLim","Gamma vs Mean 4 GLim",scatterBin,meanLow,meanHigh,scatterBin,gammaLow,gammaHigh);
  scatterPlotGammaMeanGLim.push_back(scatterPlotGammaMean4GLim);
  TH2F scatterPlotGammaMean5GLim("scatterPlotGammaMean5GLim","Gamma vs Mean 5 GLim",scatterBin,meanLow,meanHigh,scatterBin,gammaLow,gammaHigh);
  scatterPlotGammaMeanGLim.push_back(scatterPlotGammaMean5GLim);
  TH2F scatterPlotGammaMean6GLim("scatterPlotGammaMean6GLim","Gamma vs Mean 6 GLim",scatterBin,meanLow,meanHigh,scatterBin,gammaLow,gammaHigh);
  scatterPlotGammaMeanGLim.push_back(scatterPlotGammaMean6GLim);
  TH2F scatterPlotGammaMean7GLim("scatterPlotGammaMean7GLim","Gamma vs Mean 7 GLim",scatterBin,meanLow,meanHigh,scatterBin,gammaLow,gammaHigh);
  scatterPlotGammaMeanGLim.push_back(scatterPlotGammaMean7GLim);
  TH2F scatterPlotGammaMean8GLim("scatterPlotGammaMean8GLim","Gamma vs Mean 8 GLim",scatterBin,meanLow,meanHigh,scatterBin,gammaLow,gammaHigh);
  scatterPlotGammaMeanGLim.push_back(scatterPlotGammaMean8GLim);
  
  vector<TH2F> scatterPlotGammaMeanBest;
  TH2F scatterPlotGammaMean1Best("scatterPlotGammaMean1Best","Gamma vs Mean 1 Best",scatterBin,meanLow,meanHigh,scatterBin,gammaLow,gammaHigh);
  scatterPlotGammaMeanBest.push_back(scatterPlotGammaMean1Best);
  TH2F scatterPlotGammaMean2Best("scatterPlotGammaMean2Best","Gamma vs Mean 2 Best",scatterBin,meanLow,meanHigh,scatterBin,gammaLow,gammaHigh);
  scatterPlotGammaMeanBest.push_back(scatterPlotGammaMean2Best);
  TH2F scatterPlotGammaMean3Best("scatterPlotGammaMean3Best","Gamma vs Mean 3 Best",scatterBin,meanLow,meanHigh,scatterBin,gammaLow,gammaHigh);
  scatterPlotGammaMeanBest.push_back(scatterPlotGammaMean3Best);
  TH2F scatterPlotGammaMean4Best("scatterPlotGammaMean4Best","Gamma vs Mean 4 Best",scatterBin,meanLow,meanHigh,scatterBin,gammaLow,gammaHigh);
  scatterPlotGammaMeanBest.push_back(scatterPlotGammaMean4Best);
  TH2F scatterPlotGammaMean5Best("scatterPlotGammaMean5Best","Gamma vs Mean 5 Best",scatterBin,meanLow,meanHigh,scatterBin,gammaLow,gammaHigh);
  scatterPlotGammaMeanBest.push_back(scatterPlotGammaMean5Best);
  TH2F scatterPlotGammaMean6Best("scatterPlotGammaMean6Best","Gamma vs Mean 6 Best",scatterBin,meanLow,meanHigh,scatterBin,gammaLow,gammaHigh);
  scatterPlotGammaMeanBest.push_back(scatterPlotGammaMean6Best);
  TH2F scatterPlotGammaMean7Best("scatterPlotGammaMean7Best","Gamma vs Mean 7 Best",scatterBin,meanLow,meanHigh,scatterBin,gammaLow,gammaHigh);
  scatterPlotGammaMeanBest.push_back(scatterPlotGammaMean7Best);
  TH2F scatterPlotGammaMean8Best("scatterPlotGammaMean8Best","Gamma vs Mean 8 Best",scatterBin,meanLow,meanHigh,scatterBin,gammaLow,gammaHigh);
  scatterPlotGammaMeanBest.push_back(scatterPlotGammaMean8Best);
  
  
  vector<TH2F> scatterPlotGammaMeanBestSLim;
  TH2F scatterPlotGammaMean1BestSLim("scatterPlotGammaMean1BestSLim","Gamma vs Mean Best 1 SLim",scatterBin,meanLow,meanHigh,scatterBin,gammaLow,gammaHigh);
  scatterPlotGammaMeanBestSLim.push_back(scatterPlotGammaMean1BestSLim);
  TH2F scatterPlotGammaMean2BestSLim("scatterPlotGammaMean2BestSLim","Gamma vs Mean Best 2 SLim",scatterBin,meanLow,meanHigh,scatterBin,gammaLow,gammaHigh);
  scatterPlotGammaMeanBestSLim.push_back(scatterPlotGammaMean2BestSLim);
  TH2F scatterPlotGammaMean3BestSLim("scatterPlotGammaMean3BestSLim","Gamma vs Mean Best 3 SLim",scatterBin,meanLow,meanHigh,scatterBin,gammaLow,gammaHigh);
  scatterPlotGammaMeanBestSLim.push_back(scatterPlotGammaMean3BestSLim);
  TH2F scatterPlotGammaMean4BestSLim("scatterPlotGammaMean4BestSLim","Gamma vs Mean Best 4 SLim",scatterBin,meanLow,meanHigh,scatterBin,gammaLow,gammaHigh);
  scatterPlotGammaMeanBestSLim.push_back(scatterPlotGammaMean4BestSLim);
  TH2F scatterPlotGammaMean5BestSLim("scatterPlotGammaMean5BestSLim","Gamma vs Mean Best 5 SLim",scatterBin,meanLow,meanHigh,scatterBin,gammaLow,gammaHigh);
  scatterPlotGammaMeanBestSLim.push_back(scatterPlotGammaMean5BestSLim);
  TH2F scatterPlotGammaMean6BestSLim("scatterPlotGammaMean6BestSLim","Gamma vs Mean Best 6 SLim",scatterBin,meanLow,meanHigh,scatterBin,gammaLow,gammaHigh);
  scatterPlotGammaMeanBestSLim.push_back(scatterPlotGammaMean6BestSLim);
  TH2F scatterPlotGammaMean7BestSLim("scatterPlotGammaMean7BestSLim","Gamma vs Mean Best 7 SLim",scatterBin,meanLow,meanHigh,scatterBin,gammaLow,gammaHigh);
  scatterPlotGammaMeanBestSLim.push_back(scatterPlotGammaMean7BestSLim);
  TH2F scatterPlotGammaMean8BestSLim("scatterPlotGammaMean8BestSLim","Gamma vs Mean Best 8 SLim",scatterBin,meanLow,meanHigh,scatterBin,gammaLow,gammaHigh);
  scatterPlotGammaMeanBestSLim.push_back(scatterPlotGammaMean8BestSLim);
  
  vector<TH2F> scatterPlotGammaMeanBestMLim;
  TH2F scatterPlotGammaMean1BestMLim("scatterPlotGammaMean1BestMLim","Gamma vs Mean Best 1 MLim",scatterBin,meanLow,meanHigh,scatterBin,gammaLow,gammaHigh);
  scatterPlotGammaMeanBestMLim.push_back(scatterPlotGammaMean1BestMLim);
  TH2F scatterPlotGammaMean2BestMLim("scatterPlotGammaMean2BestMLim","Gamma vs Mean Best 2 MLim",scatterBin,meanLow,meanHigh,scatterBin,gammaLow,gammaHigh);
  scatterPlotGammaMeanBestMLim.push_back(scatterPlotGammaMean2BestMLim);
  TH2F scatterPlotGammaMean3BestMLim("scatterPlotGammaMean3BestMLim","Gamma vs Mean Best 3 MLim",scatterBin,meanLow,meanHigh,scatterBin,gammaLow,gammaHigh);
  scatterPlotGammaMeanBestMLim.push_back(scatterPlotGammaMean3BestMLim);
  TH2F scatterPlotGammaMean4BestMLim("scatterPlotGammaMean4BestMLim","Gamma vs Mean Best 4 MLim",scatterBin,meanLow,meanHigh,scatterBin,gammaLow,gammaHigh);
  scatterPlotGammaMeanBestMLim.push_back(scatterPlotGammaMean4BestMLim);
  TH2F scatterPlotGammaMean5BestMLim("scatterPlotGammaMean5BestMLim","Gamma vs Mean Best 5 MLim",scatterBin,meanLow,meanHigh,scatterBin,gammaLow,gammaHigh);
  scatterPlotGammaMeanBestMLim.push_back(scatterPlotGammaMean5BestMLim);
  TH2F scatterPlotGammaMean6BestMLim("scatterPlotGammaMean6BestMLim","Gamma vs Mean Best 6 MLim",scatterBin,meanLow,meanHigh,scatterBin,gammaLow,gammaHigh);
  scatterPlotGammaMeanBestMLim.push_back(scatterPlotGammaMean6BestMLim);
  TH2F scatterPlotGammaMean7BestMLim("scatterPlotGammaMean7BestMLim","Gamma vs Mean Best 7 MLim",scatterBin,meanLow,meanHigh,scatterBin,gammaLow,gammaHigh);
  scatterPlotGammaMeanBestMLim.push_back(scatterPlotGammaMean7BestMLim);
  TH2F scatterPlotGammaMean8BestMLim("scatterPlotGammaMean8BestMLim","Gamma vs Mean Best 8 MLim",scatterBin,meanLow,meanHigh,scatterBin,gammaLow,gammaHigh);
  scatterPlotGammaMeanBestMLim.push_back(scatterPlotGammaMean8BestMLim);
  
  vector<TH2F> scatterPlotGammaMeanBestGLim;
  TH2F scatterPlotGammaMean1BestGLim("scatterPlotGammaMean1BestGLim","Gamma vs Mean Best 1 GLim",scatterBin,meanLow,meanHigh,scatterBin,gammaLow,gammaHigh);
  scatterPlotGammaMeanBestGLim.push_back(scatterPlotGammaMean1BestGLim);
  TH2F scatterPlotGammaMean2BestGLim("scatterPlotGammaMean2BestGLim","Gamma vs Mean Best 2 GLim",scatterBin,meanLow,meanHigh,scatterBin,gammaLow,gammaHigh);
  scatterPlotGammaMeanBestGLim.push_back(scatterPlotGammaMean2BestGLim);
  TH2F scatterPlotGammaMean3BestGLim("scatterPlotGammaMean3BestGLim","Gamma vs Mean Best 3 GLim",scatterBin,meanLow,meanHigh,scatterBin,gammaLow,gammaHigh);
  scatterPlotGammaMeanBestGLim.push_back(scatterPlotGammaMean3BestGLim);
  TH2F scatterPlotGammaMean4BestGLim("scatterPlotGammaMean4BestGLim","Gamma vs Mean Best 4 GLim",scatterBin,meanLow,meanHigh,scatterBin,gammaLow,gammaHigh);
  scatterPlotGammaMeanBestGLim.push_back(scatterPlotGammaMean4BestGLim);
  TH2F scatterPlotGammaMean5BestGLim("scatterPlotGammaMean5BestGLim","Gamma vs Mean Best 5 GLim",scatterBin,meanLow,meanHigh,scatterBin,gammaLow,gammaHigh);
  scatterPlotGammaMeanBestGLim.push_back(scatterPlotGammaMean5BestGLim);
  TH2F scatterPlotGammaMean6BestGLim("scatterPlotGammaMean6BestGLim","Gamma vs Mean Best 6 GLim",scatterBin,meanLow,meanHigh,scatterBin,gammaLow,gammaHigh);
  scatterPlotGammaMeanBestGLim.push_back(scatterPlotGammaMean6BestGLim);
  TH2F scatterPlotGammaMean7BestGLim("scatterPlotGammaMean7BestGLim","Gamma vs Mean Best 7 GLim",scatterBin,meanLow,meanHigh,scatterBin,gammaLow,gammaHigh);
  scatterPlotGammaMeanBestGLim.push_back(scatterPlotGammaMean7BestGLim);
  TH2F scatterPlotGammaMean8BestGLim("scatterPlotGammaMean8BestGLim","Gamma vs Mean Best 8 GLim",scatterBin,meanLow,meanHigh,scatterBin,gammaLow,gammaHigh);
  scatterPlotGammaMeanBestGLim.push_back(scatterPlotGammaMean8BestGLim);
  #endif
 
/////////////////////////////////////////////////////////////////////////////////////////////////////////////
//DELTA CHI SQUARE HISTO////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////
  TH1F ToyMCDeltaChisquare("myToyLocalDeltaChiSqua  re","Toy MC Delta Chi Square Distribution",600,0,60);
  ToyMCDeltaChisquare.GetXaxis()->SetTitle("#Delta#chi^{2}  (Toy MC)");

////////////////////////////////////////////////////////////////////////////////////
  //SIGNAL FRACTIONS//
  Variable *nB = new Variable("nBkg",events,0,1E4); 
  #ifdef NEGSIG
  Variable *sFrac = new Variable("fSig",0.,0.001,-NEGSIG,NEGSIG);
  #endif
  #ifndef NEGSIG
  Variable *sFrac = new Variable("fSig",0.,0.001,0.0,0.25);
  #endif
  //Variable *sfracB = new Variable("f_best",0,0.001,0.0,1.0);


////////////////////////////////////////////////////////////////////////////////////
  //CHI SQUARES & NLL
  Double_t NullFitChiSq;

  //Nll
  Double_t NullNLL;


////////////////////////////////////////////////////////////////////////////////////
  //EVALUATION VECTORS
  vector<Double_t> ValsNull;
  vector<Double_t> ValsTot;
  vector<Double_t> ValsSig;
  vector<Variable*> vars;
  vector<PdfBase*> comps;

////////////////////////////////////////////////////////////////////////////////////
//COUNTERS
  int cycles=0; //TOTAL CYCLES
  int nfits=0;  //total number of fits
  int sectionCounter = 0;
  Double_t totalpdf=0;
  
  int problemFit[72]={9,13,31,58,87,101,102,112,119,124,139,142,150,152,173,182,205,218,224,232,235,274,316,319,324,328,332,340,355,369,374,379,405,422,440,444,479,481,505,508,509,511,516,528,555,586,627,639,646,651,653,663,679,684,685,692,706,710,763,792,797,801,812,825,837,845,875,892,911,914,926,947};
  
for(int l=0;l<iter;l++){

  cout<<endl;
  cout<<"~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"<<endl;
  cout<<"~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Iteration "<<l<<" ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"<<endl;
  cout<<"~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"<<endl;
  cout<<endl;

////////////////////////////////////////////////////////////////////////////////////
//INITIALIZE COUNTERS
  ++cycles;

#ifndef READING
////////////////////////////////////////////////////////////////////////////////////
//INITIALIZE DATA HIST
  for (int i = 0; i <= xvar->numbins+1; ++i) {
   genHist->SetBinContent(i,0);
 }
#endif
////////////////////////////////////////////////////////////////////////////////////
//SETTING STARTING VALUES for Bkg

  nB->error=0.5;
  nB->value=2500;

////////////////////////////////////////////////////////////////////////////////////
//SEED//
  struct timeval tp;
  gettimeofday(&tp,NULL);
  long int ms = tp.tv_sec * 1000 + tp.tv_usec / 1000;
  //cout<<"Milliseconds"<<ms<<endl;
////////////////////////////////////////////////////////////////////////////////////
//RANDOM GENERATOR//
  TRandom donram(ms+rndInt);

#ifdef READING   
////////////////////////////////////////////////////////////////////////////////////
//READING EVENTS 
////////////////////////////////////////////////////////////////////////////////////
//FILLING DATA HISTO
	
	//DATA TXT FILE
	//sprintf(filename,"./txt_files_imp/%d-ToyGenerated-10000/%d-ToyMCGenrated_%d.txt",section,section,sectionCounter);
	
	cout<<"==============================================================================="<<endl;
	cout<<"READING FILE : "<<filename<<endl;
	cout<<"==============================================================================="<<endl;
    
	sprintf(histoname,"genHist%d",l);
	sprintf(bufferstring,"genHist%d",l);

	genHist = (TH1F*)fileInput->Get(histoname);

	GooFile.cd();
	
	#ifdef TOYWRITE
	ToyGenFile.cd();
    genHist->Write();
    GooFile.cd();
    #endif
	
	
#else
////////////////////////////////////////////////////////////////////////////////////
//GENERATING EVENTS - HIT & MISS
  Double_t roll=0;
  Double_t background=0;
  
	#ifdef TOYWRITE
	
	//sprintf(filename,"txt_files_imp/%d-%dToyMCDeltaChisGooNLL-%f-%d.txt",iter,Clock,randomize,l);
	//ofstream toyGenerated(filename);
	
	#endif

   //#pragma omp parallel for
  for (int j = 0; j < events; ++j) {
    xvar->value = donram.Uniform(0.56)+1.008;
    background = fondo(xvar->value);
    roll = donram.Uniform(10);
    if (roll > background) {
      --j;
      continue; }

    if(isnan(background)){ //CHECKING NAN PROBLEMS
     --j;
     continue;}

    if ((xvar->value < xvar->lowerlimit) || (xvar->value > xvar->upperlimit)) {
       --j;
      continue;}

      genHist->Fill(xvar->value);
	  #ifdef TOYWRITE
	  //toyGenerated<<xvar->value;
	  #endif
  }
  
  #ifdef TOYWRITE
  
  ToyGenFile.cd();
  sprintf(bufferstring,"genHist%d",l);	
  genHist->SetTitle(bufferstring);
  genHist->SetName(bufferstring);
  genHist->Write();
  GooFile.cd();
  #endif
  
#endif
//////////////////////////////////////////////////////////////////////////////////////////////////
//FAKE TEST
//genHist->SetBinContent(2,genHist->GetBinContent(2)+500);
//genHist->SetBinContent(20,genHist->GetBinContent(20)-500);
//////////////////////////////////////////////////////////////////////////////////////////////////

//////////////////////////////////////////////////////////////////////////////////////////////////
//FILLING THE DATASETS
//////////////////////////////////////////////////////////////////////////////////////////////////

  for (int i = 1; i <= xvar->numbins; ++i) {
   dataNull.setBinContent(i-1,genHist->GetBinContent(i));
   dataSig.setBinContent(i-1,genHist->GetBinContent(i));
 }

//////////////////////////////////////////////////////////////////////////////////////////////////
//BKG PDF
//////////////////////////////////////////////////////////////////////////////////////////////////

  ThreePdf ThreePdfB("Three Bkg",xvar);
  GooPdf* ThreePdfBPtr = &ThreePdfB; 

//////////////////////////////////////////////////////////////////////////////////////////////////
//NULL PDF
//////////////////////////////////////////////////////////////////////////////////////////////////


 comps.push_back(ThreePdfBPtr);
 vars.push_back(nB);
 AddPdf NullPdf ("Three Bodies Bkg",vars,comps,1);
 GooPdf* NullPdfPtr = &NullPdf;
 vars.clear();
 comps.clear();

//////////////////////////////////////////////////////////////////////////////////////////////////
//NULL FIT
//////////////////////////////////////////////////////////////////////////////////////////////////

 NullPdfPtr->setData(&dataNull);
 NullPdfPtr->setFitControl(new BinnedNllFitInt());

 FitManager fitterNull(NullPdfPtr);
 fitterNull.fit(); nfits++; //NFITS JUST COUNTS THE NUMBER OF FITS

 fitterNull.getMinuitValues();

//////////////////////////////////////////////////////////////////////////////////////////////////
//TMINUIT NULL & MINOS
//////////////////////////////////////////////////////////////////////////////////////////////////

  TMinuit* minuNull= fitterNull.getMinuitObject();
  //minuNull->mnhess(); nfits++;  //PERFROMING HESSE
  //minuNull->mnmnos(); nfits++; //PERFROMING MINOS

  //GETTING NULL VALUES
  NullPdfPtr->evaluateAtPointsInt(xvar,ValsNull);

Double_t totalpdf = 0;

//#pragma omp parallel for
for(int k=0;k<xvar->numbins;k++){
        	
        pdfNullHist.SetBinContent(k+1,ValsNull[k]);
        totalpdf += ValsNull[k];
  	//cout<<"PDF BKG= "<<ValsNull[k]<<endl;
}
//cout<<"Total PDF= "<<totalpdf<<endl;
Double_t eventipdf=0;

//#pragma omp parallel for
for(int k=0;k<xvar->numbins;k++){

Double_t val = pdfNullHist.GetBinContent(k+1);
    val /= totalpdf;
    val *=events;
    pdfNullHist.SetBinContent(k+1, val);
    eventipdf+=val;
}

ValsNull.clear();

NullFitChiSq = chisquare(*genHist,pdfNullHist,xvar);

#ifdef DEBUGGINGCOUTS
cout<<"=================================================================================="<<endl;
cout << "CHI SQUARE BKG PDF "<<NullFitChiSq<<endl;
cout<<"=================================================================================="<<endl;
#endif

 Double_t* NullValues;
 NullValues=(minuNull->fU); //VARIABLE VALUES
 Double_t NullValues1[1];
 NullValues1[0]=NullValues[0];
 Double_t NullBkgLim=TMath::Abs(TMath::Cos(minuNull->fX[minuNull->fNiofex[0]-1])); //FSIG LIMIT
 
 NullNLL = minuNull->fAmin;

//////////////////////////////////////////////////////////////////////////////////////////////////
///// SIGNAL FIT /////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////

  //Double_t LarghStart[4]={0.005,0.0235,0.0420,0.0605};
  #ifndef GAMMAFIX
  Double_t LarghStart[4]={0.015,0.0235,0.0420,0.0605};
  #endif
  Double_t MassaStart[2]={1.040,1.050};

  Double_t SigValues[8][3];
  Double_t SigErrors[8][3];
  Double_t SignalChiSquare[8];
  Double_t SignalNLL[8];
  
  Double_t SignalLimit[8];
  Double_t MeanLimit[8];
  Double_t GammaLimit[8];

  int fitCounter = 0;

#ifndef GAMMAFIX
 for(int gscans = 0;gscans<4;gscans++){
	 
        cout<<endl;
        cout<<"~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"<<endl;
        cout<<"~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Gamma "<<gscans+1<<"("<<l<<")~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"<<endl;
#endif
#ifndef MEANFIX
		for(int mscans = 0;mscans<2;mscans++){
        cout<<"~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"<<endl;
        cout<<"~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Mass "<<mscans+1<<" ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"<<endl;
        cout<<"~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"<<endl;
        cout<<endl;
        cout<<endl;
#endif
//////////////////////////////////////////////////////////////////////////////////////////////////
//SETTING STARTING VALUES
//////////////////////////////////////////////////////////////////////////////////////////////////

  sFrac->value=0.001;
  sFrac->error=5E-4;
  
  #ifndef GAMMAFIX
  Gamma->value=LarghStart[gscans];
  Gamma->error=1E-4;
  #endif
  #ifdef GAMMAFIX
  Gamma->value=GAMMAFIX;
  int gscans = 0;
  #endif
  
  #ifndef MEANFIX
  Mean->value=MassaStart[mscans];	
  Mean->error=2E-3;
  #endif
  
  #ifdef MEANFIX
  Mean->value=MEANFIX;	
  int mscans = 0;
  #endif
 
//////////////////////////////////////////////////////////////////////////////////////////////////
//BKG PDF
//////////////////////////////////////////////////////////////////////////////////////////////////
 
 sprintf(pdfname,"Bkg Comp - %d - %d -%d",l,gscans,mscans);
 ThreePdf ThreePdfS(pdfname,xvar);
 GooPdf* ThreePdfSPtr = &ThreePdfS; 

//////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////
//SIGNAL PDF

 //GooPdf* AllPdfsSig1 = new VoigtianPdf("Bw PDF1", xvar, Mean,Sigma,Gamma);
 sprintf(pdfname,"Peak Comp - %d - %d -%d",l,gscans,mscans);
 VoigtianThreshPdf PeakPdf(pdfname, xvar, Mean,Sigma,Gamma);
 GooPdf* PeakPdfPtr = &PeakPdf;

//////////////////////////////////////////////////////////////////////////////////////////////////
 sprintf(pdfname,"Total Pdf - %d - %d -%d",l,gscans,mscans);
 AddPdf TotalPdfSig(pdfname,sFrac,PeakPdfPtr,ThreePdfSPtr,1);
 GooPdf* TotalPdfSigPtr = &TotalPdfSig; 

//////////////////////////////////////////////////////////////////////////////////////////////////
////SIGNAL FIT (MIGRAD)

 TotalPdfSigPtr->setData(&dataSig);
 TotalPdfSigPtr->setFitControl(new BinnedNllFitInt());
 FitManager fitterSig(TotalPdfSigPtr);

////////////////////////////////////////////////////////////////////////////////////
  //SIGNAL TMINUIT OBJECTS
  TMinuit* minuSig;

//////////////////////////////////////////////////////////////////////////////////////////////////
//SETTING MIGRAD FIT

  fitterSig.fit(); nfits++;
  fitterSig.getMinuitValues();
  minuSig = fitterSig.getMinuitObject();

//////////////////////////////////////////////////////////////////////////////////////////////////
//HESSE FIT
 //minuSig1->mnhess(); nfits++;

//////////////////////////////////////////////////////////////////////////////////////////////////
//TMINUIT SIGNAL
//////////////////////////////////////////////////////////////////////////////////////////////////

 Double_t* SigVal;
 Double_t* SigErr;

 SigVal=(minuSig->fU); //VARIABLE VALUES
 SigErr=(minuSig->fWerr); //VARIABLE ERRORS
 
 SigValues[fitCounter][0]=SigVal[0]; //fSig
 SigValues[fitCounter][1]=SigVal[1]; //Mean
 SigValues[fitCounter][2]=SigVal[3]; //Gamma
 
 SigErrors[fitCounter][0]= SigErr[0];
 SigErrors[fitCounter][1]= SigErr[1]; 
 SigErrors[fitCounter][2]= SigErr[3];
 
 SignalLimit[fitCounter]=TMath::Abs(TMath::Cos(minuSig->fX[minuSig->fNiofex[0]-1])); // fSig limit
 MeanLimit[fitCounter]=TMath::Abs(TMath::Cos(minuSig->fX[minuSig->fNiofex[1]-1]));   // Mean limit
 GammaLimit[fitCounter]=TMath::Abs(TMath::Cos(minuSig->fX[minuSig->fNiofex[3]-1]));  // Gamma limit
 

//////////////////////////////////////////////////////////////////////////////////////////////////
//SIGNAL PARAMETERS VALUES
//////////////////////////////////////////////////////////////////////////////////////////////////

 TotalPdfSigPtr->evaluateAtPointsInt(xvar,ValsSig);

 totalpdf = 0;

//#pragma omp parallel for
for(int k=0;k<xvar->numbins;k++){
        pdfSigHistos[fitCounter].SetBinContent(k+1,ValsSig[k]);
        totalpdf += ValsSig[k];
}

//#pragma omp parallel for
for(int k=0;k<xvar->numbins;k++){

Double_t val = pdfSigHistos[fitCounter].GetBinContent(k+1);
    val /= totalpdf;
    val *= events;
    pdfSigHistos[fitCounter].SetBinContent(k+1, val);
}
    totalpdf = 0;
    ValsSig.clear();
//////////////////////////////////////////////////////////////////////////////////////////////////
//SIG CHI SQUARE -SIG NLL
//////////////////////////////////////////////////////////////////////////////////////////////////

//#ifndef NLLBEST
SignalChiSquare[fitCounter] = chisquare(*genHist,pdfSigHistos[fitCounter],xvar);
if(abs(NullFitChiSq-SignalChiSquare[fitCounter])<=5E-5){
	double BestChi=SignalChiSquare[fitCounter];
	for(int k=0;k<8;k++) SignalChiSquare[k]=BestChi;
	cout<<"=================================================================================="<<endl;
	cout<<"==============  JUMPING =========================================================="<<endl;
	cout<<"=================================================================================="<<endl;
	break;
	
} 
//#endif
//chisquare1= genHist->Chi2Test(&pdfSig1Hist,"CHI2UW");

SignalNLL[fitCounter] = minuSig->fAmin;

#ifdef DEBUGGINGCOUTS
#ifndef NLLBEST
cout<<"=================================================================================="<<endl;
cout<<"CHI SIG CHI SQUARE = "<<SignalChiSquare[fitCounter]<<endl;
cout<<"=================================================================================="<<endl;
#endif
#endif

//////////////////////////////////////////////////////////////////////////////////////////////////
//CLEANING SIGNAL FITS

    TotalPdfSigPtr->clearCurrentFit();
    PeakPdfPtr->clearCurrentFit();
    ThreePdfSPtr->clearCurrentFit();
    minuSig->mncler();
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
	++fitCounter;

 #ifndef MEANFIX
 }

 if(abs(NullFitChiSq-SignalChiSquare[fitCounter])<=5E-5){
        double BestChi=SignalChiSquare[fitCounter];
        for(int k=0;k<8;k++) SignalChiSquare[k]=BestChi;
        cout<<"=================================================================================="<<endl;
        cout<<"==============  JUMPING =========================================================="<<endl;
        cout<<"=================================================================================="<<endl;
        break;

}

 #endif
 #ifndef GAMMAFIX
 }
 #endif

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//BEST CHI SQUARE-NLL BUFFERS & COUNTERS
Double_t bestChiSquare = -2.0;
Double_t bestChiSquareNLL = -2.0;
Double_t bestChi = -2.0;
Double_t bestNLL = 1000.0;
Double_t NLLChiSquare = 0;

int BestFitNum = 0;
int BestFitNumNLL = 0;
int BestFit = 0;
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//BEST CHI SELECTION
#ifndef NLLBEST
for (int p=0; p<fitCounter; p++) {
	
	if(bestChiSquare < NullFitChiSq-SignalChiSquare[p] && NullFitChiSq-SignalChiSquare[p]>=0){ bestChiSquare = approximate(NullFitChiSq-SignalChiSquare[p]); BestFitNum = p;}
				
	cout<<"----- Signal Chi Int "<<p+1<<" = "<<SignalChiSquare[p]<<"----- Best Chi Int = "<<bestChiSquare<<endl;
            }
	ToyMCDeltaChisquare.Fill(bestChiSquare);
	bestChi = bestChiSquare;
	BestFit = BestFitNum;
	chiFile<<bestChiSquare<<endl;//"  "<<BestFitNum<<endl;
	
	/*
	if(bestChiSquare<=1E-5){
		
		for(int j=0;j<8;j++){
			deltaChiStarts[j].Fill(approximate(NullFitChiSq-SignalChiSquare[j]));
		}
		
	}
	 */
	 
	//cout<<"Likelihood HISTOs"<<endl;
	double likeliHoodNull = 0.0;
	double likeliHoodSignal = 0.0;
	
	for(int k = 1;k<29;k++){
		double termineSig = (pdfSigHistos[BestFitNum].GetBinContent(k));
		double termineNull = pdfNullHist.GetBinContent(k);
		
		likeliHoodSignal += -2*genHist->GetBinContent(k)*log(termineSig);
		likeliHoodNull += -2*genHist->GetBinContent(k)*log(termineNull);
	}
	
	fileNLL<<likeliHoodNull-likeliHoodSignal<<endl;
	
#endif

#ifdef NLLBEST
for (int p=0; p<fitCounter; p++) {
	
	if(bestNLL > SignalNLL[p]){ bestNLL = SignalNLL[p]; BestFitNumNLL = p;}
            }
			
	NLLChiSquare = chisquare(*genHist,pdfSigHistos[BestFitNumNLL],xvar);
	bestChiSquareNLL = approximate(NullFitChiSq-NLLChiSquare);
	Double_t deltaChiNLL = bestChiSquareNLL-bestChiSquare;

	ToyMCDeltaChisquare.Fill(bestChiSquareNLL);
	
	bestChi = bestChiSquareNLL;
	BestFit = BestFitNumNLL;
	chiFileNLL<<bestChiSquareNLL<<endl; //"  "<<deltaChiNLL<<"  "<<BestFitNumNLL<<endl;
	//nullNLLFile<<NullNLL<<endl;
	//fileNLL<<bestNLL<<endl;
/*	
	if(bestChiSquareNLL<=1E-5){
		
		for(int j=0;j<8;j++){
			deltaChiStarts[j].Fill(approximate(NullFitChiSq-chisquare(*genHist,pdfSigHistos[j],xvar)));
		}
		
	}
*/	

	double likeliHoodNull = 0.0;
	double likeliHoodSignal = 0.0;
	
	for(int k = 1;k<29;k++){
		double termineSig = (pdfSigHistos[BestFitNum].GetBinContent(k));
		double termineNull = pdfNullHist.GetBinContent(k);
		
		likeliHoodSignal += -2*genHist->GetBinContent(k)*log(termineSig);
		likeliHoodNull += -2*genHist->GetBinContent(k)*log(termineNull);
	}
	
	fileNLL<<likeliHoodNull-likeliHoodSignal<<endl;
#endif

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//CHI COMPARE
/*
	Double_t maxDifference = 0.0;
	chiCompare<<"Best = "<<BestFitNum<<"List -->";
	
	for (int p=0; p<fitCounter; p++) {
		chiCompare<<p<<"  CHI =  "<<SignalChiSquare[p]<<"  NLL = "<<SignalNLL[p];
	}

	for (int p=0; p<fitCounter; p++) {
		for (int k=0; k<fitCounter; k++) 
			if(maxDifference<abs(SignalChiSquare[p]-SignalChiSquare[k])) maxDifference = SignalChiSquare[p]-SignalChiSquare[k];
			}
			 * 
	chiCompare<<" ---- Best Chi NLL  "<<BestFitNumNLL<<"  "<<NLLChiSquare<<" max diff = "<<maxDifference<<endl;
*/
	
    #ifdef STARTINGPOINTS
	#ifdef NLLBEST
	if(bestChi<=1E-5){
	startingPointsNLL->Fill(BestFitNumNLL);
	}
	#endif
	#ifndef NLLBEST
	if(bestChi<=1E-5){
	startingPoints->Fill(BestFitNum);
	}
	#endif
	#endif
	

#ifdef SCATTERPLOTS
#ifndef NLLBEST
Int_t flag = BestFitNum;
Double_t bestChiScatter = bestChiSquare;
#endif
#ifdef NLLBEST
Int_t flag = BestFitNumNLL;
Double_t bestChiScatter = bestChiSquareNLL;
#endif

	if(MeanLimit[flag]>=0.001 && GammaLimit[flag]>=0.001 && SignalLimit[flag]>=0.001){
    scatterPlotChiSigBest.Fill(SigValues[flag][0],bestChiScatter);
    scatterPlotChiGammaBest.Fill(SigValues[flag][2],bestChiScatter);
    scatterPlotChiMeanBest.Fill(SigValues[flag][1],bestChiScatter);
    scatterPlotSigMeanBest.Fill(SigValues[flag][1],SigValues[flag][0]);
    scatterPlotSigGammaBest.Fill(SigValues[flag][2],SigValues[flag][0]);
    scatterPlotGammaMeanBest[flag].Fill(SigValues[flag][1],SigValues[flag][2]);
        }

    if(SignalLimit[flag]<0.001){
    scatterPlotChiSigBestSLim.Fill(SigValues[flag][0],bestChiScatter);
    scatterPlotChiGammaBestSLim.Fill(SigValues[flag][2],bestChiScatter);
    scatterPlotChiMeanBestSLim.Fill(SigValues[flag][1],bestChiScatter);
    scatterPlotSigMeanBestSLim.Fill(SigValues[flag][1],SigValues[flag][0]);
    scatterPlotSigGammaBestSLim.Fill(SigValues[flag][2],SigValues[flag][0]);
    scatterPlotGammaMeanBestSLim[flag].Fill(SigValues[flag][1],SigValues[flag][2]);
    }else{
    if(MeanLimit[flag]<0.001){
            scatterPlotChiMeanBestMLim.Fill(SigValues[flag][1],bestChiScatter);
            scatterPlotSigMeanBestMLim.Fill(SigValues[flag][1],SigValues[flag][0]);
            scatterPlotChiSigBest.Fill(SigValues[flag][0],bestChiScatter);
            scatterPlotChiGammaBest.Fill(SigValues[flag][2],bestChiScatter);
            scatterPlotSigGammaBest.Fill(SigValues[flag][2],SigValues[flag][0]);
            scatterPlotGammaMeanBestMLim[flag].Fill(SigValues[flag][1],SigValues[flag][2]);
    }else{
    if(GammaLimit[flag]<0.001){
            scatterPlotChiMeanBest.Fill(SigValues[flag][1],bestChiScatter);
            scatterPlotSigMeanBest.Fill(SigValues[flag][1],SigValues[flag][0]);
            scatterPlotChiSigBest.Fill(SigValues[flag][0],bestChiScatter);
            scatterPlotChiGammaBestGLim.Fill(SigValues[flag][2],bestChiScatter);
            scatterPlotSigGammaBestGLim.Fill(SigValues[flag][2],SigValues[flag][0]);
            scatterPlotGammaMeanBestGLim[flag].Fill(SigValues[flag][1],SigValues[flag][2]);
    }
    }}

#endif

#ifdef GOODPLOTS
////////////////////////////////////////////////////////////////////////////////////
//PLOTTING GOOD SIGNAL FIT
if(bestChi>=GOODPLOTS){
//////////////////////////////////////////////////////////////////////////////////////////////////
////SAVING TOYMC DISTRIBUTION
//////////////////////////////////////////////////////////////////////////////////////////////////
	sprintf(bufferstring,"ToyMC Good %d",l);	
	genHist->SetTitle(bufferstring);
	genHist->SetName(bufferstring);
	genHist->Write();
	sprintf(bufferstring,"genHist");	
	genHist->SetTitle(bufferstring);
	genHist->SetName(bufferstring);
//////////////////////////////////////////////////////////////////////////////////////////////////
////SAVING PARAMETERS
//////////////////////////////////////////////////////////////////////////////////////////////////
	//paramFile<<SigValues[BestFit][0]<<endl;
	//paramFile<<SigValues[BestFit][1]<<endl;
	//paramFile<<SigValues[BestFit][2]<<endl;
	//paramFile<<endl;
    }
#endif

//////////////////////////////////////////////////////////////////////////////////////////////////
//CLEANING BKG FITS

NullPdfPtr->clearCurrentFit();
ThreePdfBPtr->clearCurrentFit();
 ++sectionCounter;
}//COMPLETE CYCLE

  stopCPU = times(&stopProc);
  gettimeofday(&stopTime, NULL);

#ifdef SCATTERPLOTS

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//CHI & SIG SCATTER PLOTS MEAN BEST/////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

//CHI vs SIG
  scatterPlotChiSigBest.SetFillColor(1);
  scatterPlotChiSigBestSLim.SetFillColor(2);
  scatterPlotChiSigBest.Write();
  scatterPlotChiSigBestSLim.Write();
  
  scatterPlotChiSigBestSLim.SetStats(0);
  scatterPlotChiSigBest.Draw("box");
  scatterPlotChiSigBestSLim.Draw("box same");

  sprintf(canvasname,"Chi - Signal Fraction Best");	
  canvas->SetTitle(canvasname);
  canvas->SetName(canvasname);
  /*sprintf(canvasname,"plots/ScatterChiSignalBest.png",k+1);
  canvas->SaveAs(canvasname);*/
  canvas->Write(); 
  canvas->Clear();

//CHI vs MEAN
	scatterPlotChiMeanBest.SetFillColor(1);
   scatterPlotChiMeanBestSLim.SetFillColor(2);
   scatterPlotChiMeanBestMLim.SetFillColor(3);
   scatterPlotChiMeanBestGLim.SetFillColor(4);
   scatterPlotChiMeanBestSLim.Write();
   scatterPlotChiMeanBestMLim.Write();
   scatterPlotChiMeanBestGLim.Write();
   scatterPlotChiMeanBest.Write();
   
   scatterPlotChiMeanBestSLim.SetStats(0);
   scatterPlotChiMeanBestMLim.SetStats(0);
   scatterPlotChiMeanBestGLim.SetStats(0);
   scatterPlotChiMeanBest.Draw("box");
   scatterPlotChiMeanBestSLim.Draw("box same");
   scatterPlotChiMeanBestMLim.Draw("box same");
   scatterPlotChiMeanBestGLim.Draw("box same");
   
   sprintf(canvasname,"Chi - Mean Best");	
   canvas->SetTitle(canvasname);
  canvas->SetName(canvasname);
  /*sprintf(canvasname,"plots/ScatterChiMeanBest.png",k+1);
  canvas->SaveAs(canvasname);*/
  canvas->Write(); 
  canvas->Clear();
   
//CHI vs GAMMA
	scatterPlotChiGammaBest.SetFillColor(1);
   scatterPlotChiGammaBestSLim.SetFillColor(2);
   scatterPlotChiGammaBestMLim.SetFillColor(3);
   scatterPlotChiGammaBestGLim.SetFillColor(4);
   scatterPlotChiGammaBest.Write();
   scatterPlotChiGammaBestSLim.Write();
   scatterPlotChiGammaBestMLim.Write();
   scatterPlotChiGammaBestGLim.Write();
   
   scatterPlotChiGammaBestSLim.SetStats(0);
   scatterPlotChiGammaBestMLim.SetStats(0);
   scatterPlotChiGammaBestGLim.SetStats(0);
   scatterPlotChiGammaBest.Draw("box");
   scatterPlotChiGammaBestSLim.Draw("box same");
   scatterPlotChiGammaBestMLim.Draw("box same");
   scatterPlotChiGammaBestGLim.Draw("box same");
   
   sprintf(canvasname,"Chi - Gamma Best");	
   canvas->SetTitle(canvasname);
  canvas->SetName(canvasname);
  /*sprintf(canvasname,"plots/ScatterChiGammaBest.png",k+1);
  canvas->SaveAs(canvasname);*/
  canvas->Write(); 
  canvas->Clear();
   
//SIG vs MEAN
   scatterPlotSigMeanBest.SetFillColor(1);
   scatterPlotSigMeanBestSLim.SetFillColor(2);
   scatterPlotSigMeanBestMLim.SetFillColor(3);
   scatterPlotSigMeanBest.Write();
   scatterPlotSigMeanBestSLim.Write();
   scatterPlotSigMeanBestMLim.Write();
   
   scatterPlotSigMeanBestSLim.SetStats(0);
   scatterPlotSigMeanBestMLim.SetStats(0);
   scatterPlotSigMeanBest.Draw("box");
   scatterPlotSigMeanBestSLim.Draw("box same");
   scatterPlotSigMeanBestMLim.Draw("box same");
   sprintf(canvasname,"Signal Fraction - Mean Best");	
   canvas->SetTitle(canvasname);
  canvas->SetName(canvasname);
  /*sprintf(canvasname,"plots/ScatterSignalMeanBest.png",k+1);
  canvas->SaveAs(canvasname);*/
  canvas->Write(); 
  canvas->Clear();
  
//SIG vs GAMMA
   scatterPlotSigGammaBest.SetFillColor(1);
   scatterPlotSigGammaBestSLim.SetFillColor(2);
   scatterPlotSigGammaBestGLim.SetFillColor(4);
   scatterPlotSigGammaBest.Write();
   scatterPlotSigGammaBestSLim.Write();
   scatterPlotSigGammaBestGLim.Write();
   
   scatterPlotSigGammaBestSLim.SetStats(0);
   scatterPlotSigGammaBestGLim.SetStats(0);
   scatterPlotSigGammaBest.Draw("box");
   scatterPlotSigGammaBestSLim.Draw("box same");
   scatterPlotSigGammaBestGLim.Draw("box same");
   sprintf(canvasname,"Signal Fraction - Gamma Best");
	
   canvas->SetTitle(canvasname);
  canvas->SetName(canvasname);
  /*sprintf(canvasname,"plots/ScatterSignalGammaBest.png");
  canvas->SaveAs(canvasname);*/
  canvas->Write(); 
  canvas->Clear();

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//GAMMA MEAN BEST///////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//GAMMA MEAN TOTAL BEST
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  
  for(int k=0;k<8;k++){
  scatterPlotGammaMeanBestGLim[k].SetFillColor(4);
  scatterPlotGammaMeanBestMLim[k].SetFillColor(3);
  scatterPlotGammaMeanBestSLim[k].SetFillColor(2);
  scatterPlotGammaMeanBest[k].SetFillColor(1);
  
  scatterPlotGammaMeanBestGLim[k].Write();
  scatterPlotGammaMeanBest[k].Write();
  scatterPlotGammaMeanBestSLim[k].Write();
  scatterPlotGammaMeanBest[k].Write();
  
  scatterPlotGammaMeanBestGLim[k].SetStats(0);
  scatterPlotGammaMeanBestMLim[k].SetStats(0);
  scatterPlotGammaMeanBestSLim[k].SetStats(0);

  scatterPlotGammaMeanBestGLim[k].Draw("box");
  scatterPlotGammaMeanBest[k].Draw("box same");
  scatterPlotGammaMeanBestSLim[k].Draw("box same");
  scatterPlotGammaMeanBest[k].Draw("box same");
  
  grid->Draw();
	
   sprintf(canvasname,"Gamma-Mean Best (%d)",k+1);	
  canvas->SetTitle(canvasname);
  canvas->SetName(canvasname);
  /*sprintf(canvasname,"plots/ScatterGammaMeanBest%d.png",k+1);
  canvas->SaveAs(canvasname);*/
  canvas->Write(); 
  canvas->Clear();
}
  
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//GAMMA MEAN GOOD BEST
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  
  for(int k=0;k<8;k++){
	  scatterPlotGammaMeanBest[k].SetFillColor(1);
	  scatterPlotGammaMeanBest[k].Draw("box");
	  scatterPlotGammaMeanBest[k].Write();
	   grid->Draw();
	   sprintf(canvasname,"Gamma-Mean Best Good (%d)",k+1);	
		canvas->SetTitle(canvasname);
		canvas->SetName(canvasname);
	   /*sprintf(canvasname,"plots/ScatterGammaMeanBestGood%d.png",k+1);
	     canvas->SaveAs(canvasname);*/
		 canvas->Write(); canvas->Clear();
	  
  }


////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//GAMMA MEAN SLIM BEST
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

for(int k=0;k<8;k++){
	  scatterPlotGammaMeanBestSLim[k].SetFillColor(2);
	  scatterPlotGammaMeanBestSLim[k].Draw("box");
	  scatterPlotGammaMeanBestSLim[k].Write();
	   grid->Draw();
	   sprintf(canvasname,"Gamma-Mean Best SLim(%d)",k+1);	
		canvas->SetTitle(canvasname);
		canvas->SetName(canvasname);
	   /*sprintf(canvasname,"plots/ScatterGammaMeanBestSLim%d.png",k+1);
	     canvas->SaveAs(canvasname);*/
		 canvas->Write(); canvas->Clear();
	  
  }
	
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//GAMMA MEAN MLIM BEST
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
for(int k=0;k<8;k++){
	  scatterPlotGammaMeanBestMLim[k].SetFillColor(2);
	  scatterPlotGammaMeanBestMLim[k].Draw("box");
	  scatterPlotGammaMeanBestMLim[k].Write();
	   grid->Draw();
	   sprintf(canvasname,"Gamma-Mean Best MLim(%d)",k+1);	
		canvas->SetTitle(canvasname);
		canvas->SetName(canvasname);
	   /*sprintf(canvasname,"plots/ScatterGammaMeanBestMLim%d.png",k+1);
	     canvas->SaveAs(canvasname);*/
		 canvas->Write(); canvas->Clear();
	  
  }
	
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//GAMMA MEAN GLIM BEST
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

for(int k=0;k<8;k++){
	  scatterPlotGammaMeanBestMLim[k].SetFillColor(4);
	  scatterPlotGammaMeanBestMLim[k].Draw("box");
	  scatterPlotGammaMeanBestMLim[k].Write();
	  grid->Draw();
	  sprintf(canvasname,"Gamma-Mean Best MLim(%d)",k+1);	
	  canvas->SetTitle(canvasname);
	  canvas->SetName(canvasname);
	 /*sprintf(canvasname,"plots/ScatterGammaMeanBestMLim%d.png",k+1);
	     canvas->SaveAs(canvasname);*/
	  canvas->Write(); canvas->Clear();
	  
  }
  
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#endif
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#ifdef STARTINGPOINTS
 //startingPointsNLL->Draw();
 /*sprintf(canvasname,"plots/%d.%d-StartingPointsNll--%dIters-Log.png",Date,Clock,iter);
 canvas->SaveAs(canvasname);*/
 //canvas->Write(); canvas->Clear();
 
 startingPoints->Draw();
 sprintf(canvasname,"plots/%d.%d-StartingPoints--%dIters-Log.png",Date,Clock,iter);
 canvas->SaveAs(canvasname);
 //canvas->Write(); canvas->Clear();
#endif
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//CHI HISTO LOG/////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  canvas->SetLogy(1);
  //GooFile.cd();
  ToyMCDeltaChisquare.SetMarkerStyle(8);
  ToyMCDeltaChisquare.SetMarkerSize(0.4);
  ToyMCDeltaChisquare.Draw();
  ToyMCDeltaChisquare.SetLineColor(kGreen);
  ToyMCDeltaChisquare.Draw("lsame");
  
  sprintf(canvasname,"plots/%d.%d-DeltaChis--%dIters-Log.eps",Date,Clock,iter);
  canvas->SaveAs(canvasname);
  canvas->Clear();
  
  //ToyMCDeltaChisquare.Write();
  /*
  canvas->Divide(4,2);
  for(int h=0;h<8;h++){
	  canvas->cd(h+1);
	  deltaChiStarts[h].SetLineColor(h+1);
	  deltaChiStarts[h].Draw("LTEXT45");
	  deltaChiStarts[h].Write();
  }
  sprintf(canvasname,"plots/%d.%d-DeltaChisStarts--%dIters-Log.eps",Date,Clock,iter);
  canvas->SaveAs(canvasname);
  canvas->Clear();
  
  deltaChiStarts[0].Draw();
	  
  for(int h=1;h<8;h++){
	  deltaChiStarts[h].Draw("same");
  }
  
  sprintf(canvasname,"plots/%d.%d-DeltaChisStartsCompare--%dIters-Log.eps",Date,Clock,iter);
  canvas->SaveAs(canvasname);
  canvas->Clear();
  */

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  //sprintf(filename,"txt_files_imp/%d/%d/%d%d%d/%d-ToyMCDeltaChisGoox%d%d%d-%d-%d-%d.txt",Date,iter,gFix,mFix,sNeg,iter,gFix,mFix,sNeg,Date,Clock,toys);
  ofstream timeFile("./Times/time.txt",std::ofstream::app);
  
  cout<<endl;
  cout<<"=================================================================================="<<endl;
  cout<<"Total cycles            "<< cycles <<endl;
  cout<<"Number of fits          "<<nfits<<endl;
  Double_t myCPUc = (stopCPU - startCPU)*10000;
  cout<<"Computation time:       " << (myCPUc / CLOCKS_PER_SEC) << endl ;
  cout<<"=================================================================================="<<endl;
 timeFile<<(myCPUc / CLOCKS_PER_SEC)<<std::endl;
      return 0;
}
