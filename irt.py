
import numpy as np
import math
D=1.7
# n student m item
n=1000
m=1000 
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
	return x if x>=0 else -x

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
	num_step=0
	eps=1e-6
	li=[]
	if u.sum()==u.shape[0]:return 0.0,0.0,-3.00
	if u.sum()==0:return 0.0,0.0,3.00
	for i in range(k):
		ha,hab,hb=1,0,1
		num_init=0
		# Hession Matrix negtive define for a0,b0
		while (not is_negtive_define(ha,hab,hb)) and num_init<100:
			num_init+=1
			ea=2*np.random.random_sample()
			eb=6*np.random.random_sample()-3
			# lh=likehood(theta,u,ea,eb,c)
			
			xa,xb=getDerivative(theta,u,ea,eb,c)
			ha,hab,hb=getHess(theta,u,ea,eb,c)

			det=ha*hb-hab**2
			hi=1/det
		lh=likehood(theta,u,ea,eb,c)
		deltaxa=hi*(hb*xa-hab*xb)
		deltaxb=hi*(-hab*xa+ha*xb)
		while abs(xa)>=eps and abs(xb)>=eps and num_step<100:
			num_step+=1
			miu=1.0
			tempa=ea-miu*deltaxa
			tempb=eb-miu*deltaxb
			ha,hab,hb=getHess(theta,u,tempa,tempb,c)
			# newtown down hill method 
			while tempa<0 or tempa>2 or tempb<-3 or tempb>3 or (not is_negtive_define(ha,hab,hb)) or lh>likehood(theta,u,tempa,tempb,c):
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

# init theta0 is fixed
def esttheta(a,b,c,u):
	global D
	# print u
	num_step=0
	if u.sum()==u.shape[0]:return 3.00
	if u.sum()==0:return -3.00
	etheta=math.log(float(u.sum())/(u.shape[0]-u.sum()))
	dx=1.00
	eps=1e-6
	# print 0,etheta,abs(dx),abs(dx)>eps
	while abs(dx)>eps and num_step<30:
		# print 'in while'
		num_step+=1
		tb=etheta-b
		p=c+(1-c)/(1+math.e**(-D*a*tb))
		d=D*a*(u-p)*(p-c)/(1-c)/p
		dd=D**2*a**2*(p-c)*(u*c-p**2)*(1-p)/(p**2*(1-c)**2)

		dx=d.sum()
		ddxx=dd.sum()
		# print ddxx

		etheta-=dx/ddxx
	if etheta>3.00:etheta=3.00
	if etheta<-3.00:etheta=-3.00

	return etheta

# 
def esttheta2(a,b,c,u,k=100):
	global D
	# print u
	
	if u.sum()==u.shape[0]:return 3.00
	if u.sum()==0:return -3.00
	res=[]
	for i in range(k):
		num_step=0
		etheta=6*np.random.random_sample()-3
		dx=1.00
		eps=1e-6
		# print 0,etheta,abs(dx),abs(dx)>eps
		while abs(dx)>eps and num_step<300:
			# print 'in while'
			num_step+=1
			tb=etheta-b
			p=c+(1-c)/(1+math.e**(-D*a*tb))
			d=D*a*(u-p)*(p-c)/(1-c)/p
			dd=D**2*a**2*(p-c)*(u*c-p**2)*(1-p)/(p**2*(1-c)**2)

			dx=d.sum()
			ddxx=dd.sum()
			# print ddxx
			if abs(ddxx)<eps:break

			etheta-=dx/ddxx
		if etheta>3.00:etheta=3.00
		if etheta<-3.00:etheta=-3.00
		lh=likehood(etheta,u,a,b,c)
		res.append((lh,etheta))
		# print etheta,lh
	res.sort()
	return res[-1][-1]


# random choice 
def makechoice(k):
	global m,n
	stu_item={i:np.random.choice(m, k, replace=False) for i in range(n)}
	item_stu={i:[] for i in range(m)}
	for stu in stu_item:
		for item in stu_item[stu]:
			item_stu[item].append(stu)
	for item in item_stu:
		if len(item_stu[item])<1:del item_stu[item]
	return stu_item,item_stu

# each student select k items randomly,est theta,a,b for each student,select item
def estall2(k):
	global a,b,c,y,theta
	stu_item,item_stu=makechoice(k)
	etheta=[]
	# est theta
	for i in range(len(stu_item)):
		print 'est theta' ,i
		etheta.append(esttheta(a[stu_item[i]],b[stu_item[i]],c,y[i,stu_item[i]]))

	
	for i in range(len(etheta)):
		if abs(etheta[i])>2.999:
			for item in item_stu:
				if i in item_stu[item]:item_stu[item].remove(i)
	etheta=np.array(etheta)
	# est a,b
	eas=[]
	ebs=[]
	for item in item_stu:
		print  'est ab' ,item
		# print etheta[item_stu[item]]
		# print y[item_stu[item],item]
		lh,ea,eb=estab(etheta[item_stu[item]],y[item_stu[item],item],100)	
		eas.append(ea)
		ebs.append(eb)
	eas=np.array(eas)
	ebs=np.array(ebs)
	print np.abs(etheta-theta).mean(),np.abs(eas-a).mean(),np.abs(ebs-b).mean()
	return etheta,eas,ebs

# each student select first k student ,est theta,use the ested theta est a,b for left items
def estall3(k):
	global a,b,c,y,theta,n,m
	etheta=[]
	# est theta
	for i in range(n):
		print 'est theta' ,i
		etheta.append(esttheta(a[:k],b[:k],c,y[i,:k]))
	
	etheta=np.array(etheta)
	# est a,b
	eas=[]
	ebs=[]
	for i in range(m-k):
		item=k+i
		print  'est ab' ,item
		# print etheta[item_stu[item]]
		# print y[item_stu[item],item]
		lh,ea,eb=estab(etheta,y[:,item],100)	
		eas.append(ea)
		ebs.append(eb)
	eas=np.array(eas)
	ebs=np.array(ebs)
	print np.abs(etheta-theta).mean(),np.abs(eas-a[k:]).mean(),np.abs(ebs-b[k:]).mean()
	return etheta,eas,ebs


# each student select k items randomly from first m/2 items,est theta,
# each last m/2 items select k student randomly for est a,b
def estall4(n_item=100,n_stu=100):
	global a,b,c,y,theta,n,m
	etheta=[]
	# est theta
	for i in range(n):
		print 'est theta' ,i
		indicate=np.random.choice(m/2, n_item, replace=False)
		etheta.append(esttheta(a[indicate],b[indicate],c,y[i,indicate]))
	
	etheta=np.array(etheta)
	# est a,b
	eas=[]
	ebs=[]
	for i in range(m/2):
		item=m/2+i
		print  'est ab' ,item
		# print etheta[item_stu[item]]
		# print y[item_stu[item],item]
		indicate=np.random.choice(n, n_stu, replace=False)
		lh,ea,eb=estab(etheta[indicate],y[indicate,item],100)	
		eas.append(ea)
		ebs.append(eb)
	eas=np.array(eas)
	ebs=np.array(ebs)
	print np.abs(etheta-theta).mean(),np.abs(eas-a[m/2:]).mean(),np.abs(ebs-b[m/2:]).mean()
	return etheta,eas,ebs

def estalltheta(k):
	global a,b,c,y,n,m,theta
	etheta=[]
	item=np.random.choice(m, k, replace=False)
	for i in range(n):
		etheta.append(esttheta2(a[item],b[item],c,y[i,item],200))
	etheta=np.array(etheta)
	print etheta
	print (((etheta - theta)**2).sum()/n)**0.5,np.abs(etheta - theta).mean()


	etheta=[]
	item=np.random.choice(m, k, replace=False)
	for i in range(n):
		etheta.append(esttheta(a[item],b[item],c,y[i,item]))
	etheta=np.array(etheta)
	print (((etheta - theta)**2).sum()/n)**0.5,np.abs(etheta - theta).mean()
	return etheta











