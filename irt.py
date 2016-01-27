
import numpy as np
import math
D=1.7
# 500students 30 items
n=100
m=1 
theta=np.random.normal(0,1,n)
a=np.random.lognormal(0,0.3,m)
b=np.random.normal(0,1,m)
# a=2*np.random.random_sample(m) 
# b=6*np.random.random_sample(m)-3
c=0.25
y=np.zeros((n,m))
ea=None
eb=None
all2=None

def getp(c,x):
	global D
	return c+(1.0-c)/(1.0+math.e**-x)

def getpmatrix(theta,ea,eb,c):
	p=np.zeros((theta.shape[0],ea.shape[0]))
	for i in range(theta.shape[0]):
		for j in range(ea.shape[0]):
			p[i,j]=c+(1.0-c)/(1.0+math.pow(math.e,-D*ea[j]*(theta[i]-eb[j])))
	return p

getp3=np.vectorize(getp,otypes=[np.float])
# score matrix
# for i in range(n):
# 	for j in range(m):
# 		y[i,j]=1 if np.random.random_sample()<getp(theta[i],a[j],b[j],c) else 0

def makeabtheta():
	global theta
	global a
	global b
	global n
	global m
	theta=np.random.normal(0,1,n)
	a=np.random.lognormal(0,0.3,m)
	b=np.random.normal(0,1,m)

def makey():
	global y
	p=getp3(c,D*a*(theta.reshape((-1,1))-b.reshape((1,-1))))
	u=np.random.random_sample((n,m))
	y[u<p]=1
	y[u>=p]=0
	# print y.sum(axis=0).max(),y.sum(axis=0).min()


def est2(n):
	global theta
	global c
	global y
	global D
	global a
	global b
	global ea
	global eb
	error=1.0

	# ea=np.zeros((a.shape[0],))+1
	# eb=np.zeros((a.shape[0],))
	ea=np.random.lognormal(0,0.3,m)
	eb=np.random.normal(0,1,m)

	while error>0.0001:
		p=getp3(c,D*ea*(theta.reshape((-1,1))-eb.reshape((1,-1))))
		tb=(theta.reshape((-1,1))-eb.reshape((1,-1)))
		da=D*tb*(p-c)*(y-p)/p/(1-c)
		db=-D*ea*(p-c)*(y-p)/p/(1-c)

		ddaa=(D/(1-c))**2*tb**2*(p-c)*(1-p)/p*(y*c/p-p)
		ddbb=(D*ea/(1-c))**2*(p-c)*(1-p)/p*(y*c/p-p)
		ddab=-D/(1-c)*(p-c)*((y/p-1)+D*ea/(1-c)*tb*(1-p)/p*(y*c/p-p))

		xa=da[:n,:].sum(axis=0)
		xb=db[:n,:].sum(axis=0)
		ha=ddaa[:n,:].sum(axis=0)
		hab=ddab[:n,:].sum(axis=0)
		hb=ddbb[:n,:].sum(axis=0)

		# print xa
		# print xb
		# print ea
		# print eb

		
		hi=1/(ha*hb-hab**2)
		# print hi
		deltaxa=hi*(hb*xa-hab*xb)
		deltaxb=hi*(-hab*xa+ha*xb)
		
		ea-=deltaxa
		eb-=deltaxb

		error=(deltaxa**2).sum()+(deltaxb**2).sum()
		# print error
		try:
			lh=likehood(theta,ea,eb,c,y)
		except:
			return -10000,(((a-ea)**2).sum()/a.shape[0])**0.5,(((b-eb)**2).sum()/b.shape[0])**0.5
		print 'error=',error,'loglikehood=',lh
	print 'real loglikehood=',likehood(theta,a,b,c,y)
	errora=(((a-ea)**2).sum()/a.shape[0])**0.5
	errorb=(((b-eb)**2).sum()/b.shape[0])**0.5
	print 'error a=',errora,'error b=',errorb
	# return ea,eb
	return lh,errora,errorb

def abs(x):
	return x if x<=0 else -x

def likehood(theta,u,a,b,c):
	global D
	tb=theta-b
	# print theta,u,a,b,c
	p=c+(1-c)/(1+math.e**(-D*a*tb))
	lh=u*np.log(p)+(1-u)*np.log(1-p)
	return lh.sum()
	

def estall():
	global n
	global all2
	all2=[]
	for i in range(10):
		makeabtheta()
		makey()
		res=[]
		for j in range(10):
			res.append(est2(n))
		res.sort()
		all2.append(res[-1])
	return np.array(all2)


def getDerivative(theta,u,a,b,c):
	global D
	tb=theta-b
	p=c+(1-c)/(1+math.e**(-D*a*tb))
	da=D*tb*(p-c)*(u-p)/p/(1-c)
	db=-D*a*(p-c)*(u-p)/p/(1-c)
	xa=da.sum()
	xb=db.sum()
	return xa,xb

def getHess(theta,u,a,b,c):
	global D
	tb=theta-b
	p=c+(1-c)/(1+math.e**(-D*a*tb))
	ddaa=(D/(1-c))**2*tb**2*(p-c)*(1-p)/p*(u*c/p-p)
	ddbb=(D*a/(1-c))**2*(p-c)*(1-p)/p*(u*c/p-p)
	ddab=-D/(1-c)*(p-c)*((u/p-1)+D*a/(1-c)*tb*(1-p)/p*(u*c/p-p))
	ha=ddaa.sum()
	hab=ddab.sum()
	hb=ddbb.sum()
	return ha,hab,hb

def is_negtive_define(ha,hab,hb):
	det=ha*hb-hab**2
	a=True if ha<0 and det>0 else False
	return a

def estab(theta,u,k):
	global c
	eps=1e-6
	li=[]
	for i in range(k):
		ha,hab,hb=1,0,1
		# Hession Matrix negtive define for a0,b0
		while not is_negtive_define(ha,hab,hb):
			ea=2*np.random.random_sample()
			eb=6*np.random.random_sample()-3
			lh=likehood(theta,u,ea,eb,c)
			
			xa,xb=getDerivative(theta,u,ea,eb,c)
			ha,hab,hb=getHess(theta,u,ea,eb,c)

			det=ha*hb-hab**2
			hi=1/det
		
		deltaxa=hi*(hb*xa-hab*xb)
		deltaxb=hi*(-hab*xa+ha*xb)
		while abs(xa)>=eps and abs(xb)>=eps:
			miu=1.0
			tempa=ea-miu*deltaxa
			tempb=eb-miu*deltaxb
			ha,hab,hb=getHess(theta,u,tempa,tempb,c)
			# newtown down hill method 
			while (not is_negtive_define(ha,hab,hb)) or lh>likehood(theta,u,tempa,tempb,c):
				miu*=0.5
				tempa=ea-miu*deltaxa
				tempb=eb-miu*deltaxb
				ha,hab,hb=getHess(theta,u,tempa,tempb,c)

			ea=tempa
			eb=tempb
			xa,xb=getDerivative(theta,u,ea,eb,c)
			det=ha*hb-hab**2
			hi=1/det
			deltaxa=hi*(hb*xa-hab*xb)
			deltaxb=hi*(-hab*xa+ha*xb)
			lh=likehood(theta,u,tempa,tempb,c)
		# print lh
		li.append((lh,ea,eb))
	li.sort()
	return li[-1]

def test():
	global c
	global theta
	global y
	global a
	global b
	t=estab(theta,y.reshape(-1,),100)
	rlh=likehood(theta,y.reshape(-1,),a,b,c)
	print 'est ',t
	print 'real ',rlh,a,b

def testall(num):
	global c
	global theta
	global y
	global a
	global b
	ts=[]
	rs=[]
	for i in range(num):
		makeabtheta()
		makey()
		t=estab(theta,y.reshape(-1,),100)[1:]
		ts.append(t)
		rs.append([a,b])
	ts=np.array(ts)
	rs=np.array(rs).reshape(-1,2)
	print ((ts-rs)**2).sum(axis=0)/num
	return ts,rs









