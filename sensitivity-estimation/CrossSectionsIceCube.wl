(* ::Package:: *)

BeginPackage["CrossSectionsIceCube`"]

\[CapitalGamma]::usage="decay width GeV"
\[Tau]DLS::usage="lifetime in restframe in s"
\[Sigma]IWW::usage="total cross";
\[Sigma]IWWy::usage="total cross";
\[Sigma]WW\[Psi]::usage="total cross";
\[Sigma]WW::usage="total cross";
\[Sigma]IWW\[Psi]::usage="total cross";

d\[Sigma]dxWW::usage="cross section";
d\[Sigma]dxETL::usage="cross section";
d\[Sigma]dxIWW::usage="cross section";

d\[Sigma]dyWW::usage="cross section";
d\[Sigma]dyIWW::usage="cross section ";
d\[Sigma]d\[Psi]WW::usage="cross section";
d\[Sigma]d\[Psi]IWW::usage="cross section";

d\[Sigma]dxdcosWW::usage="cross section";
d\[Sigma]dxdcosWWcorrect\[Chi]::usage="cross section";
d\[Sigma]dydcosWW::usage="cross section";
d\[Sigma]dxdcosIWW::usage="cross section";
d\[Sigma]dydcosIWW::usage="cross section";
d\[Sigma]dydcosIWWnoprefactor::usage="cross section";
d\[Sigma]dxdcosWWsimplified::usage="hello";
d\[Sigma]dxdcosETL::usage="hello axel";

uWWdisplay::usage="";
\[Chi]WWdisplay::usage="";
\[Chi]WWdisplayCorrect::usage="";
t::usage="";

Begin["Private`"]

(* BEGIN, experimental parameters *)
m\[Mu]=\!\(TraditionalForm\`0.1056583745\);
me=\!\(TraditionalForm\`0.000510999461\);
mi=m\[Mu];
mf=me;
\[Alpha]=1/137;
\[Theta]kmax = 0.1;
\[Psi]max = 3.0;
xmin:=ma/E0;
xmax:=1-mf/E0-ma^4/(8*E0^3*A); (* include nucleus recoil *)
ymin := 1-xmax;
ymax := 1 - xmin;
(* initialize variables that later will be changed in the computation loop *)
ma = 0.3;
h=1.;
E0=160.;
\[CapitalDelta]m2:=ma^2-mi^2-mf^2;

(* form factor *)
M=16.;
A=16.;
Z = 8.;
aF = Z^(-1/3)*111/me;
dF = 0.164*A^(-2/3);
F2[t_]:=Z^2*(aF^2*t/(1+aF^2*t))^2*(1/(1+t/dF))^2

(* analytical photon flux *)
(* equation 26 in PRD104.076012*)
\[Chi]tilde[t_,tmin_]:=td^2/((ta-td)^3)*( 
(ta-td)*(ta+tmin)/(t+ta)
+(ta-td)*(td+tmin)/(t+td)
+(ta+td+2*tmin)*Log[(t+td)/(t+ta)]
);
\[Chi]analytical[tmin_,tmax_]:=Z^2*(\[Chi]tilde[tmax,tmin]-\[Chi]tilde[tmin,tmin]);
td=dF;
ta=(1/aF)^2;

(* END, experimental parameters and caclualtion configuration *)
(* ------------------------------------------------- *)

(* ------------------------------------------------- *)
(* BEGIN, variable definitions *)
p := Sqrt[E0^2-mi^2];
k[x_]:=Sqrt[x^2*E0^2-ma^2];
u[x_,\[Theta]k_]:=ma^2+mi^2-mf^2-2*x*E0^2+2*p*k[x]*Cos[\[Theta]k];
uWW[x_,\[Theta]k_]:=-x*E0^2*\[Theta]k^2-ma^2*(1-x)/x+mi^2*(1-x)-mf^2; (* this is u at tmin *)
uWWdisplay[x_,\[Theta]k_,E00_,m_,hh_]:=Module[{},
	ma=m;
	E0=E00;
	h=hh;
	uWW[x,\[Theta]k]
]
beta[y_]:=Sqrt[1.0-mf^2/(y^2*E0^2)];
t2[y_,\[Psi]_] := -y*E0^2*\[Psi]^2-mf^2*(1.0-y)/y+mi^2(1.0-y);
t[y_,\[Psi]_] := ma^2-t2[y,\[Psi]];

V[x_,\[Theta]k_]:=Sqrt[p^2+k[x]^2-2*p*k[x]*Cos[\[Theta]k]];
q0[t_]:=-t/(2M);
q[t_]:=Sqrt[t+q0[t]^2];
cos\[Theta]q[x_,\[Theta]k_,t_] := ((E0*(1-x)+q0[t])^2 - V[x,\[Theta]k]^2-q[t]^2 -mf^2)/(2*V[x,\[Theta]k]*q[t]);
sin\[Theta]q[x_,\[Theta]k_,t_] := Sqrt[1.0-cos\[Theta]q[x,\[Theta]k,t]^2];

(* bounds for t-integral *)
tOfQ[Q_]:=2*M(Sqrt[M^2+Re[Q^2]]-M);
tmin[x_,\[Theta]k_]:=Module[{Qmin},
Qmin = (V[x,\[Theta]k]*(u[x,\[Theta]k]+2*(EpPlusEf[x])*M)-EpPlusEf[x]*Sqrt[u[x,\[Theta]k]^2+4*M*EpPlusEf[x]*u[x,\[Theta]k]+4*M^2*V[x,\[Theta]k]^2])/(2*EpPlusEf[x]^2-2*V[x,\[Theta]k]^2);
tOfQ[Qmin]
];
tmax[x_,\[Theta]k_]:=Module[{Qmax},
Qmax = (V[x,\[Theta]k]*(u[x,\[Theta]k]+2*(EpPlusEf[x])*M)+EpPlusEf[x]*Sqrt[u[x,\[Theta]k]^2+4*M*EpPlusEf[x]*u[x,\[Theta]k]+4*M^2*V[x,\[Theta]k]^2])/(2*EpPlusEf[x]^2-2*V[x,\[Theta]k]^2);
tOfQ[Qmax]
];
EpPlusEf[x_]:=E0*(1-x)+M; (* used in bounds for t-integral *)

(* END, variable definitions *)
(* ------------------------------------------------- *)

(* ------------------------------------------------- *)
(* BEGIN, cross section definitions *)

(* TOTAL \[Sigma] *)
\[Sigma]WW[E00_,m_,hh_]:=Module[{},
ma=m;
E0=E00;
h=hh;
NIntegrate[d\[Sigma]dxdcosWW[x,\[Theta]k,E0,ma,h],{\[Theta]k,0,\[Theta]kmax},{x,xmin,xmax}]
]

\[Sigma]IWW[E00_,m_,hh_]:=Module[{integral},
ma=m;
E0=E00;
h=hh;
NIntegrate[d\[Sigma]dxIWW[x,E0,ma,h],{x,xmin,xmax}]
]

\[Sigma]WWy[E00_,m_,hh_]:=Module[{integral},
ma=m;
E0=E00;
h=hh;
NIntegrate[d\[Sigma]dydcosWW[y,\[Psi],E0,ma,h],{\[Psi],0,\[Psi]max},{y,ymin,ymax}]
]

\[Sigma]IWWy[E00_,m_,hh_]:=Module[{integral},
ma=m;
E0=E00;
h=hh;
NIntegrate[d\[Sigma]dyIWW[y,E0,ma,h],{y,ymin,ymax}]
]

\[Sigma]WW\[Psi][E00_,m_,hh_]:=Module[{integral},
ma=m;
E0=E00;
h=hh;
NIntegrate[d\[Sigma]d\[Psi]WWintegrand[y,\[Psi]],{\[Psi],0,\[Psi]max},{y,ymin,ymax}]
]

\[Sigma]IWW\[Psi][E00_,m_,hh_]:=Module[{integral},
ma=m;
E0=E00;
h=hh;
NIntegrate[d\[Sigma]d\[Psi]IWWintegrand[y,\[Psi]],{\[Psi],0,\[Psi]max},{y,ymin,ymax}]
]

(* D\[Sigma]DX *)

(* WW *)
d\[Sigma]dxWW[x_,E00_,m_,hh_]:=Module[{},

NIntegrate[d\[Sigma]dxdcosWW[x,\[Theta]k,E00,m,hh],{\[Theta]k,0,\[Theta]kmax}]
]
d\[Sigma]dxdcosWW[x_,\[Theta]k_,E00_,m_,hh_]:=Module[{A2},
	ma=m;
	E0=E00;
	h=hh;
	A2 =(\[CapitalDelta]m2*(ma^2*(1 - x) +
     x*(mf^2 + mi^2*(x - 1)) + uWW[x,\[Theta]k]*x))/uWW[x,\[Theta]k]^2 + 
  x^2/(2 - 2*x);

	prefactorWW[x]*Sin[\[Theta]k]*\[Chi]WW[x,\[Theta]k]*A2/(uWW[x,\[Theta]k]^2)
]

d\[Sigma]dxdcosWWcorrect\[Chi][x_,\[Theta]k_,E00_,m_,hh_]:=Module[{A2},
	ma=m;
	E0=E00;
	h=hh;
	A2 =(\[CapitalDelta]m2*(ma^2*(1 - x) + 
     x*(mf^2 + mi^2*(x - 1)) + uWW[x,\[Theta]k]*x))/uWW[x,\[Theta]k]^2 + 
  x^2/(2 - 2*x);
	prefactorWW[x]*Sin[\[Theta]k]*\[Chi]WWdisplayCorrect[x,\[Theta]k,E0,ma,h]*A2/(uWW[x,\[Theta]k]^2)
]

\[Chi]WW[x_,\[Theta]k_]:=\[Chi]analytical[uWW[x,\[Theta]k]^2/(4*E0^2*(1-x)^2),E0*E0];
(*\[Chi]WW[x_,\[Theta]k_]:=\[Chi]analytical[uWW[x,\[Theta]k]^2/(4*E0^2*(1-x)^2),ma^2+m\[Mu]^2];*)

\[Chi]WWdisplay[x_,\[Theta]k_,E00_,m_,hh_]:=Module[{},
	ma=m;
	E0=E00;
	h=hh;
	\[Chi]WW[x,\[Theta]k]
]
\[Chi]WWdisplayCorrect[x_,\[Theta]k_,E00_,m_,hh_]:=Module[{},
	ma=m;
	E0=E00;
	h=hh;
	\[Chi]analytical[uWW[x,\[Theta]k]^2/(4*E0^2*(1-x)^2),ma^2+mi^2]
]

prefactorWW[x_]:=h^2*\[Alpha]^2/(2\[Pi])*k[x]*E0*(1-x);

d\[Sigma]dxdcosWWsimplified[x_,\[Theta]k_,E00_,m_,hh_]:=Module[{},
	ma=m;
	E0=E00;
	h=hh;
	
	h^2*\[Alpha]^2/(2\[Pi])*Sin[\[Theta]k]*k[x]*E0*\[Chi]WW[x,\[Theta]k]*((1-x)*(\[CapitalDelta]m2*(ma^2*(1 - x) + 
     x*(mf^2-mi^2 + mi^2*x) + uWW[x,\[Theta]k]*x))/uWW[x,\[Theta]k]^4 + 
  x^2/(2 (uWW[x,\[Theta]k]^2)))
	
]

(* ---------------------------------------------------- *)

(* IWW *)
d\[Sigma]dxdcosIWW[x_,\[Theta]k_,E00_,m_,hh_]:=Module[{A2},
	ma=m;
	E0=E00;
	h=hh;
	A2 =(\[CapitalDelta]m2*(ma^2*(1 - x) + 
     x*(mf^2 + mi^2*(x - 1)) + uWW[x,\[Theta]k]*x))/uWW[x,\[Theta]k]^2 + 
  x^2/(2 - 2*x);
	prefactorWW[x]*Sin[\[Theta]k]*A2/(uWW[x,\[Theta]k]^2)
]

d\[Sigma]dxIWW[x_,E00_,m_,hh_]:=Module[{integral,umin,umax},
ma=m;
E0=E00;
h=hh;

umin = uWW[x,\[Theta]kmax];
umax = uWW[x,0];
integral= d\[Sigma]dxIWWantiderivative[x,umax] - d\[Sigma]dxIWWantiderivative[x,umin];
prefactorIWW[x]*\[Chi]IWW[x]*integral
]
d\[Sigma]dxIWWantiderivative[x_,uval_] := (\[CapitalDelta]m2*(ma^2*x - ma^2 - mf^2*x - mi^2*x^2 + mi^2*x))/(3*uval^3) - (\[CapitalDelta]m2*x)/(2*uval^2) + x^2/(2*uval*(x - 1));

\[Chi]IWW[x_]:=\[Chi]analytical[ma^4/(4*E0^2),E0*E0];
prefactorIWW[x_]:= h^2*\[Alpha]^2/(4\[Pi])*Sqrt[1-ma^2/(x^2*E0^2)]*(1-x);
(* ---------------------------------------------------- *)

(* ---------------------------------------------------- *)

(* ETL \[Phi] NUMERICAL *)
d\[Sigma]dxETL[x_,E00_,m_,hh_]:=Module[{},
	ma=m;
	E0=E00;
	h=hh;
	NIntegrate[d\[Sigma]dxdcosETLintegrand[x,\[Theta]k,t,\[Phi]],{\[Theta]k,0,\[Theta]kmax},{t,tmin[x,\[Theta]k],tmax[x,\[Theta]k]},{\[Phi],0,2\[Pi]}]
];

prefactorETL[x_]:=h^2/(4*\[Pi])*\[Alpha]^2*k[x]*E0/p/(8*M^2);

d\[Sigma]dxdcosETL[x_,\[Theta]k_,E00_,m_,hh_]:=Module[{},
	ma=m;
	E0=E00;
	h=hh;
	NIntegrate[d\[Sigma]dxdcosETLintegrand[x,\[Theta]k,t,\[Phi]],{t,tmin[x,\[Theta]k],tmax[x,\[Theta]k]},{\[Phi],0,2\[Pi]}]
];

d\[Sigma]dxdcosETLintegrand[x_,\[Theta]k_,t_,\[Phi]_]:=Module[{},
	prefactorETL[x]*Sin[\[Theta]k]/V[x,\[Theta]k]*F2[t]/t^2*A2[x,\[Theta]k,t,\[Phi]]/(2*\[Pi])
];

qk[x_,\[Theta]k_,t_,\[Phi]_]:=q[t]*k[x]/V[x,\[Theta]k]*(p(cos\[Theta]q[x,\[Theta]k,t]*Cos[\[Theta]k]+sin\[Theta]q[x,\[Theta]k,t]*Sin[\[Theta]k]*Cos[\[Phi]])-k[x]*cos\[Theta]q[x,\[Theta]k,t]);
qp[x_,\[Theta]k_,t_,\[Phi]_]:=q[t]*p/V[x,\[Theta]k]*(p*cos\[Theta]q[x,\[Theta]k,t]-k[x]*(cos\[Theta]q[x,\[Theta]k,t]*Cos[\[Theta]k]-sin\[Theta]q[x,\[Theta]k,t]*Sin[\[Theta]k]*Cos[\[Phi]]));

PP[t_]:=4*M^2+t;
Pk[x_,\[Theta]k_,t_,\[Phi]_]:=(2*M+t/(2*M))*x*E0+qk[x,\[Theta]k,t,\[Phi]];
Pp0[x_,\[Theta]k_,t_,\[Phi]_]:=(2*M+t/(2*M))*E0+qp[x,\[Theta]k,t,\[Phi]];
Pp1[x_,\[Theta]k_,t_,\[Phi]_]:=Pp0[x,\[Theta]k,t,\[Phi]]-Pk[x,\[Theta]k,t,\[Phi]];
s[x_,\[Theta]k_,t_,\[Phi]_]:=-(1+E0/M)*t-2*qp[x,\[Theta]k,t,\[Phi]];

A2[x_,\[Theta]k_,t_,\[Phi]_] := 1/2*(
(s[x,\[Theta]k,t,\[Phi]]+u[x,\[Theta]k])^2/(s[x,\[Theta]k,t,\[Phi]]*u[x,\[Theta]k])*PP[t]
-4t (Pk[x,\[Theta]k,t,\[Phi]]^2)/(s[x,\[Theta]k,t,\[Phi]]*u[x,\[Theta]k])
+((s[x,\[Theta]k,t,\[Phi]]+u[x,\[Theta]k])/(s[x,\[Theta]k,t,\[Phi]]*u[x,\[Theta]k]))^2*\[CapitalDelta]m2*(t*PP[t]-4((u[x,\[Theta]k]*Pp0[x,\[Theta]k,t,\[Phi]]+s[x,\[Theta]k,t,\[Phi]]*Pp1[x,\[Theta]k,t,\[Phi]])/(s[x,\[Theta]k,t,\[Phi]]+u[x,\[Theta]k]))^2))
(* ---------------------------------------------------- *)

(* WW dsdy *)
d\[Sigma]dydcosWW[y_,\[Psi]_,E00_,m_,hh_]:=Module[{\[Chi],tmin,tmax},
	ma=m;
	E0=E00;
	h=hh;
	(* photon flux *)
	tmin = (t[y,\[Psi]]/(2*E0*(1-y)))^2;
	(*tmax = mi^2+ma^2;*)
	tmax = E0*E0;
	\[Chi]=\[Chi]analytical[tmin,tmax];

	h^2*\[Chi]*Sin[\[Psi]](E0^2*(\[Alpha]^2/(2\[Pi]))*(1 - y)^3*beta[y]*(1/(2*t[y,\[Psi]]^2) - \[CapitalDelta]m2/(t[y,\[Psi]]^3) + (\[CapitalDelta]m2*(mf^2 + y*(mi^2*y + \[CapitalDelta]m2)))/(t[y,\[Psi]]^4*y)))
]
d\[Sigma]dydcosIWWnoprefactor[y_,\[Psi]_,E00_,m_,hh_]:=Module[{\[Chi],tmin,tmax},
	ma=m;
	E0=E00;
	h=hh;

	Sin[\[Psi]]*(1. - y)^3*beta[y]*(1.0/(2.0*t[y,\[Psi]]^2) - \[CapitalDelta]m2/(t[y,\[Psi]]^3) + (\[CapitalDelta]m2*(mf^2 + y*(mi^2*y + \[CapitalDelta]m2)))/(t[y,\[Psi]]^4*y))
]

d\[Sigma]dydcosIWW[y_,\[Psi]_,E00_,m_,hh_]:=Module[{\[Chi],tmin,tmax},
	ma=m;
	E0=E00;
	h=hh;
	(* photon flux *)
	tmin = ma^4/(4*E0^2);
	(*tmax = mi^2+ma^2;*)
	tmax = E0*E0;
	\[Chi]=\[Chi]analytical[tmin,tmax];
	h^2*\[Chi]*Sin[\[Psi]]*(E0^2*(\[Alpha]^2/(2\[Pi]))*(1. - y)^3*beta[y]*(1.0/(2.0*t[y,\[Psi]]^2) - \[CapitalDelta]m2/(t[y,\[Psi]]^3) + (\[CapitalDelta]m2*(mf^2 + y*(mi^2*y + \[CapitalDelta]m2)))/(t[y,\[Psi]]^4*y)))
]

d\[Sigma]dyWW[y_,E00_,m_,hh_]:=Module[{},
	NIntegrate[d\[Sigma]dydcosWW[y,\[Psi],E00,m,hh],{\[Psi],0,\[Psi]max}]
]
(* ------------------------------------------------- *)
(* WW dsdpsi *)
d\[Sigma]d\[Psi]WWintegrand[y_,\[Psi]_]:=Module[{\[Chi],tmin,tmax},

	(* photon flux *)
	tmin = (t[y,\[Psi]]/(2*E0*(1-y)))^2;
	(*tmax = mi^2+ma^2;*)
	tmax = E0*E0;
	\[Chi]=\[Chi]analytical[tmin,tmax];
	h^2*Sin[\[Psi]]*(E0^2*(\[Alpha]^2/(2\[Pi]))*(1 - y)^3*beta[y]*(1/(2*t[y,\[Psi]]^2) - \[CapitalDelta]m2/(t[y,\[Psi]]^3) + (\[CapitalDelta]m2*(mf^2 + y*(mi^2*y + \[CapitalDelta]m2)))/(t[y,\[Psi]]^4*y))*\[Chi])
]

d\[Sigma]d\[Psi]WW[\[Psi]_,E00_,m_,hh_]:=Module[{},
	ma=m;
	E0=E00;
	h=hh;
NIntegrate[d\[Sigma]d\[Psi]WWintegrand[y,\[Psi]],{y,ymin, ymax}]
]
(* ------------------------------------------------- *)
(* IWW dsdy *)
d\[Sigma]dyIWWantiderivative[y_,tVal_]:= -1/(2*tVal)+\[CapitalDelta]m2/(2*tVal^2) - \[CapitalDelta]m2*(mf^2+y(mi^2*y+\[CapitalDelta]m2))/(3*y*tVal^3)

d\[Sigma]dyIWW[y_,E00_,m_,hh_]:=Module[{tmin,tmax,\[Chi],tildetmax,tildetmin,integral},
	ma=m;
	E0=E00;
	h=hh;
(* photon flux *)
tmin = ma^4/(4*E0^2);
	(*tmax = mi^2+ma^2;*)
	tmax = E0*E0;
\[Chi]=\[Chi]analytical[tmin,tmax];

(* analytical integral over \tilde{t} *)
tildetmax = t[y,\[Psi]max];
tildetmin = t[y,0];

integral = d\[Sigma]dyIWWantiderivative[y,tildetmax] - d\[Sigma]dyIWWantiderivative[y,tildetmin];

\[Alpha]^2*h^2/(4\[Pi])*beta[y]*\[Chi]*((1-y)^3)/y*integral
]
(* ------------------------------------------------- *)
(* IWW dsd\[Psi] *)
d\[Sigma]d\[Psi]IWWintegrand[y_,\[Psi]_]:=Module[{\[Chi],tmin,tmax},
	(* photon flux *)
	tmin = ma^4/(4*E0^2);
	(*tmax = mi^2+ma^2;*)
	tmax = E0*E0;
	\[Chi]=\[Chi]analytical[tmin,tmax];
	h^2*\[Chi]*Sin[\[Psi]]*(E0^2*(\[Alpha]^2/(2\[Pi]))*(1. - y)^3*beta[y]*(1.0/(2.0*t[y,\[Psi]]^2) - \[CapitalDelta]m2/(t[y,\[Psi]]^3) + (\[CapitalDelta]m2*(mf^2 + y*(mi^2*y + \[CapitalDelta]m2)))/(t[y,\[Psi]]^4*y)))
]

d\[Sigma]d\[Psi]IWW[\[Psi]_,E00_,m_,hh_]:=Module[{},
	ma=m;
	E0=E00;
	h=hh;
NIntegrate[d\[Sigma]d\[Psi]IWWintegrand[y,\[Psi]],{y,ymin, ymax}]
]
(* ------------------------------------------------- *)

(* END, cross section definitons *)

\[CapitalGamma][m_,h_]:=Module[{momentumElectron},
momentumElectron = Sqrt[(m^2 - (me+m\[Mu])^2)*(m^2-(me-m\[Mu])^2)]/(2*m);
h^2/(8*\[Pi])*momentumElectron*(1 - (me^2+m\[Mu]^2)/m^2)
]

\[Tau]DLS[m_,h_]:=6.582*10^(-25)*1/\[CapitalGamma][m,h];

(* ------------------------------------------------- *)
(* ------------------------------------------------- *)

End[]
EndPackage[]
