import json
import random

class Crossing():
    def  __init__(self, cross:list):
        assert len(cross)==4
        self.crossing = cross
    
    def corresponceRotationPath(self,path,rotation=1,indexNext=0):# index next for twisted crossings
        return self.crossing[(list(self.crossing).index(path,indexNext)+rotation)%4]

    def __contains__(self, el):
        return el in self.crossing
    
    def renamePathLocally(self,oldPath,newPath):
        self.crossing[self.crossing.index(oldPath)]=newPath
        if oldPath not in self.crossing: return # test for twist
        self.crossing[self.crossing.index(oldPath)]=newPath

    def isPathTwisted(self,path): # when path is connected 2 times with crossing => twist
        return self.crossing.count(path)==2
    
    def isTwisted(self):
        return self.isPathTwisted(self.crossing[0]) or self.isPathTwisted(self.crossing[1])
    
    def __str__(self):
        return str(self.crossing)
    
    def up(self):
        return self.crossing[1]
    def down(self):
        return self.crossing[3]
    def right(self):
        return self.crossing[0]
    def left(self):
        return self.crossing[2]
    
    def flip(self): # flips crossing to coresponding crossing
        self.crossing[1],self.crossing[3]=self.crossing[3],self.crossing[1]
        self.crossing[0],self.crossing[2]=self.crossing[2],self.crossing[0]
    
    def invert(self): # exchanges upper and lower path
        self.crossing[1],self.crossing[2],self.crossing[3],self.crossing[0] = self.crossing

    def pathGoesOver(self,path): # does not account for twisted paths
        return self.crossing.index(path) in {1,3} # 1,3 means th path goes over the other path
    def __hash__(self) -> int:
        return (hash(self.crossing[0])+1)^(hash(self.crossing[1])-1)^(hash(self.crossing[2])+1)^(hash(self.crossing[3])-1)+1


class Knot():
    def __init__(self,k=[]):
            #[[1,5,2,4],[3,1,4,6],[5,3,6,2]]
        self.crossings=[Crossing(crossing) for crossing in k]
        self._cache={} # cache for facing paths algorithm
        self.loggs=[]

    def crossingArr(self)->list:
        return [cross.crossing for cross in self.crossings]

    def findCrossingWithPath(self,path,exclude=[]):
        return list(filter(lambda x: path in x and x not in exclude,self.crossings))

    def twist(self,path): # first reidermeister move
        assert not self.isAdjointTwisted(path)
        first_crossing,second_crossing=filter(lambda x: path in x,self.crossings)
        first_crossing.renamePathLocally(path,f"{path}-1") # replace index to split path
        second_crossing.renamePathLocally(path,f"{path}-2")
        newKnot = Crossing([f"{path}-2",f"{path}-3",f"{path}-3",f"{path}-1"]) # new twist
        self.crossings.append(newKnot)

    def untwist(self,path):
        crossing = self.findCrossingWithPath(path)
        assert len(crossing)==1
        [crossing] = crossing # unpacking
        for neighborPath in crossing.crossing: self.renamePathGlobally(neighborPath, path) # rename to path
        del self.crossings[self.crossings.index(crossing)]

    def isTwisted(self,path):
        return len(self.findCrossingWithPath(path))==1
    
    def isAdjointTwisted(self,path): # tests if the given path is adjoint to an twist (or is twisted itself)
        if self.isTwisted(path): return True

        crossing1, crossing2 = self.findCrossingWithPath(path)
        return crossing1.isTwisted() or crossing2.isTwisted()

    def poke(self,path1,path2): # poke path2 over path 1
        assert  path1 !=path2
        if path2 in self.facingPaths(path1):
            cycle = self.facingPaths(path1)
        elif self.facingPaths(path1,rotation=-1):
            cycle = self.facingPaths(path1,rotation=-1)
        else:
            assert False
        assert not self.isTwisted(path1)
        assert not self.isTwisted(path2)
        firstLeftCrossing,firstRightCrossing = self.findCrossingWithPath(path1)
        secondLeftCrossing,secondRightCrossing = self.findCrossingWithPath(path2)

        for path in cycle: # well ordering such that left and right are correct
            if path in secondRightCrossing:
                secondLeftCrossing,secondRightCrossing=secondRightCrossing,secondLeftCrossing
                break
            elif path in firstLeftCrossing:
                break
        firstLeftCrossing.renamePathLocally(path1,f"{path1}-1")
        firstRightCrossing.renamePathLocally(path1,f"{path1}-2")
        secondLeftCrossing.renamePathLocally(path2,f"{path2}-1")
        secondRightCrossing.renamePathLocally(path2,f"{path2}-2")
        newLeftCrossing = Crossing([f"{path1}-3",f"{path2}-3",f"{path1}-1",f"{path2}-1"])
        newRightCrossing = Crossing([f"{path1}-2",f"{path2}-3",f"{path1}-3",f"{path2}-2"])
        self.crossings.append(newLeftCrossing)
        self.crossings.append(newRightCrossing)
    
    def renamePathGlobally(self,oldPath,newPath):
        [
            crossing.renamePathLocally(oldPath, newPath)
            for crossing in self.findCrossingWithPath(oldPath)
        ]
    def unpoke(self,upperPath,lowerPath):
        assert upperPath!=lowerPath
        if upperPath in self.facingPaths(lowerPath):
            cycle = self.facingPaths(lowerPath)
        elif self.facingPaths(lowerPath,rotation=-1):
            cycle = self.facingPaths(lowerPath,rotation=-1)
        else:
            assert False
        assert len(cycle)==2
        leftCrossing,rightCrossing = self.findCrossingWithPath(lowerPath) # === self.findCrossingWithPath(upperPath)

        if leftCrossing.up()==rightCrossing.up()==upperPath: # order crossings to execute algo with more ezz
            pass
        elif leftCrossing.down()==rightCrossing.down()==upperPath:
            leftCrossing.flip()
            rightCrossing.flip()
        elif leftCrossing.up()==rightCrossing.down()==upperPath:
            rightCrossing.flip()
        elif leftCrossing.down()==rightCrossing.up()==upperPath:
            leftCrossing.flip()
        else: 
            assert False
        assert leftCrossing.up()==rightCrossing.up()==upperPath
        self.renamePathGlobally(rightCrossing.right(),lowerPath)
        self.renamePathGlobally(leftCrossing.left(),lowerPath)
        self.renamePathGlobally(rightCrossing.left(),lowerPath)
        self.renamePathGlobally(leftCrossing.right(),lowerPath)

        self.renamePathGlobally(rightCrossing.down(),upperPath)
        self.renamePathGlobally(leftCrossing.down(),upperPath)
        del self.crossings[self.crossings.index(leftCrossing)]
        del self.crossings[self.crossings.index(rightCrossing)] # crossings now get removed

    def slideUnder(self,underPath, rotationSide=1):
        triangleSides = set(self.facingPaths(underPath,rotation=rotationSide))
        #print("----------------------")
        #print(triangleSides)
        assert len(triangleSides)==3

        path1, path2 = triangleSides.difference({underPath})
        edges = set(self.findCrossingWithPath(path1) + self.findCrossingWithPath(path2))
        #print(self.findCrossingWithPath(path1), self.findCrossingWithPath(path2))
        #print(edges)
        assert len(edges)==3
        [underEdge] = [edge for edge in edges if underPath not in edge] # edge on oposite side of underPath (assuming slide under possible)
        
        #print(self,edges)
        leftEdge, rightEdge = edges.difference({underEdge})
    
        # assert not underEdge.isTwisted()
        # assert not rightEdge.isTwisted()
        # assert not leftEdge.isTwisted()
        
        if not leftEdge.up() in [path1,path2]: leftEdge.flip() # orienting paths for convinience (does nothing to the information)
        if not rightEdge.up() in [path1,path2]: rightEdge.flip()
        if not leftEdge.right()==underPath:
            assert rightEdge.right()==underPath
            leftEdge, rightEdge = rightEdge, leftEdge
        assert leftEdge.right()==underPath
        assert rightEdge.left()==underPath # reassuring under path is an undergoing path
        # esteblish path names for convinience
        leftPath = leftEdge.up()
        rightPath = rightEdge.up()
        highRightPath = underEdge.corresponceRotationPath(rightPath,rotation=2) # continouation of the right path (is now on left side bcs crossing)
        highLeftPath = underEdge.corresponceRotationPath(leftPath, rotation=2) # path on the mirror side of the left path
        lowerLeftPath = leftEdge.down()
        lowerRightPath = rightEdge.down()
        leftUnderPath = leftEdge.left()
        rightUnderPath = rightEdge.left()
        # actuall sliding

        self.crossings[self.crossings.index(leftEdge)].renamePathLocally(leftPath,highRightPath)
        self.crossings[self.crossings.index(leftEdge)].renamePathLocally(lowerLeftPath,rightPath)
        
        self.crossings[self.crossings.index(rightEdge)].renamePathLocally(rightPath,highLeftPath)
        self.crossings[self.crossings.index(rightEdge)].renamePathLocally(lowerRightPath,leftPath)
        
        self.crossings[self.crossings.index(underEdge)]=Crossing([lowerRightPath,leftPath,rightPath,lowerLeftPath])
        # self.crossings.append(newUnderEdge)

    def facingPaths(self,path,rotation:int=1):
        # caching
        cashHash = hash(self)^hash(path)^hash(rotation)
        if hash(self)^hash(path)^hash(rotation) in self._cache.keys():
            return self._cache[cashHash]
        # algorithm
        paths = [path]
        nextCrossing = self.findCrossingWithPath(path)[0] # choice between doesent matter
        path = nextCrossing.corresponceRotationPath(path,rotation)
        # print(self,path)
        while not path in paths:
            paths.append(path)
            if nextCrossing.isPathTwisted(path): # crosses itself through twist
                if set(nextCrossing.crossing).issubset(set(paths)):
                    break # no new paths can be found if a twist is infront of the end
                path = list(filter(lambda x: x not in paths,nextCrossing.crossing))[0] #finds only path not already scanned => next path
                paths.append(path)
            [nextCrossing] = self.findCrossingWithPath(path,exclude=[nextCrossing])
            path = nextCrossing.corresponceRotationPath(path,rotation)
        # caching & return
        self._cache[cashHash]=paths
        return paths

    def getAllPaths(self):
        paths = []
        for crossing in self.crossings:
            for path in crossing.crossing:
                if path not in paths:
                    paths.append(path)
                    yield path
    def __hash__(self):
        result = 1
        for crossing in self.crossings:
            result ^= hash(crossing)
        return result

    def reidermeisterMove(self,path,move:str,path2=None)->(bool):
        if move=="slide":
            #cycle = self.facingPaths(path)
            if self.hasThreeFacingPathsFast(path,1):
                try:
                    self.slideUnder(path,1)
                except AssertionError:
                    pass
                else:
                    self.loggs.append({"move":"slide","path":path,"face":1})
                    self._cache={}
                    return True
            #cycle = self.facingPaths(path,-1)
            if self.hasThreeFacingPathsFast(path,-1):
                try:
                    self.slideUnder(path,-1)
                except AssertionError:
                    return False
                else:
                    self.loggs.append({"move":"slide","path":path,"face":-1})
                    self._cache={}
                    return True
            return False

        if move=="untwist":
            try:
                self.untwist(path)
            except AssertionError:
                return False
            else:
                self.loggs.append({"move":"untwist","path":path})
                self._cache={}
                return True
        if move=="twist":
            try:
                self.twist(path)
            except AssertionError:
                return False
            else:
                self.loggs.append({"move":"twist","path":path})
                self._cache={}
                return True
        if move=="poke":
            assert path != path2 != None
            try:
                self.poke(path,path2)
            except AssertionError:
                return False
            else:
                self.loggs.append({"move":"poke","path":path,"path2":path2})
                self._cache={}
                return True
        if move=="unpoke":
            cycle = self.facingPaths(path)
            if not len(cycle)==2:
                cycle = self.facingPaths(path,-1)
            if len(cycle)!=2: 
                return False
            try:
                self.unpoke(cycle[0],cycle[1])
            except AssertionError:
                pass
            else:
                self.loggs.append({"move":"unpoke","path":path,"face":1})
                self._cache={}
                return True
            try:
                self.unpoke(cycle[1],cycle[0])
            except AssertionError:
                return False
            else:
                self.loggs.append({"move":"unpoke","path":path,"face":-1})
                self._cache={}
                return True

    def simpliefy(self,maxIterations=20):
        for i in range(maxIterations):
            print(i)
            self.unPokeAll()
            self.untwistAll()
            self.slideIfPossible()
            print(i,1)


    def untwistAll(self):
        wasReducible = True
        while wasReducible:
            wasReducible = False
            for path in self.getAllPaths():
                status = self.reidermeisterMove(path,"untwist")
                if status:
                    wasReducible = True
                    break

    def unPokeAll(self):
        wasReducible = True
        while wasReducible:
            wasReducible = False
            for path in self.getAllPaths():
                status = self.reidermeisterMove(path,"unpoke")
                if status:
                    wasReducible = True
                    break

    def slideIfPossible(self):
        wasReducible = True
        pathsSlided = [] # to avoid infinite loop through sliding the same paths
        while wasReducible:
            wasReducible = False
            for path in self.getAllPaths():
                if not path in pathsSlided:
                    status = self.reidermeisterMove(path,"slide")
                    if status:
                        pathsSlided.append(path)
                        wasReducible = True
                        break
    
    def slideAll(self):
        slidablePaths = self.getSlidablePaths()
        for path in slidablePaths:
            self.reidermeisterMove(path,"slide")
    
    def slidesubset(self,subsetSizePercent=1):
        slidablePaths = self.getSlidablePaths()
        slidablePaths = slidablePaths[0:int(subsetSizePercent*len(slidablePaths))]
        for path in slidablePaths:
            self.reidermeisterMove(path,"slide")

    def getSlidablePaths(self):
        return [
            path
            for path in self.getAllPaths() 
            if 
                self.hasThreeFacingPathsFast(path,1)
                or
                self.hasThreeFacingPathsFast(path,-1)
        ]

    
    def hasThreeFacingPathsFast(self,path,rotation:int=1): # not best function name / : Tests if there are more than 3 facing paths is used for acceleration of slideAll algorithm
        # algorithm
        paths = [path]
        nextCrossing = self.findCrossingWithPath(path)[0] # choice between doesent matter
        path = nextCrossing.corresponceRotationPath(path,rotation)
        while not path in paths:
            paths.append(path)
            if len(paths)>3: return False
            if nextCrossing.isPathTwisted(path): # crosses itself through twist
                if set(nextCrossing.crossing).issubset(set(paths)):
                    break # no new paths can be found if a twist is infront of the end
                path = list(filter(lambda x: x not in paths,nextCrossing.crossing))[0] #finds only path not already scanned => next path
                paths.append(path)
            if len(paths)>3: return False
            [nextCrossing] = self.findCrossingWithPath(path,exclude=[nextCrossing])
            path = nextCrossing.corresponceRotationPath(path,rotation)
        if len(paths)==3: return True
        return False


    def __str__(self):
        return json.dumps(self.dump())

    def invert(self):
        for crossing in self.crossings:
            crossing.invert()

    def annotate(self,annotation):
        for path in self.getAllPaths():
            self.renamePathGlobally(path,f"{path}.{annotation}")
    
    def __add__(self, other):
        pass

    #@functools.cached_property
    def paths(self):
        return set(
                [crossing.up() for crossing in self.crossings]
            ).union(
                set([crossing.right() for crossing in self.crossings])
            ).union(
                set([crossing.left() for crossing in self.crossings])
            ).union(
                set([crossing.down() for crossing in self.crossings])
            )

    def relableKnot(self):
        for i,path in enumerate(self.paths()):
            self.renamePathGlobally(path,f"_{i}")
        for i,path in enumerate(self.facingPaths(self.crossings[0].up(),2)): # the facing paths are such that you walk along the knot starting at an arbitrary path
            self.renamePathGlobally(path,i)

    def dump(self):
        return [crossing.crossing for crossing in self.crossings]

    def __eq__(self, __o: object) -> bool:
        self.relableKnot()
        __o.relableKnot()
        a = [tuple(crossing) for crossing in self.dump()] # assuming that the knots are relabled
        b = [tuple(crossing) for crossing in __o.dump()] # assuming that __o are the crossings
        if len(a)!=len(b):
            return False
        for i in range(0, len(self.crossings)):
            # print(np.array_equal(a,(b+i)%(2*len(self.crossings))),a,(b+i)%(2*len(self.crossings)))
            b = [
                (
                    (crossing[0]+1)%(2*len(self.crossings)),
                    (crossing[1]+1)%(2*len(self.crossings)),
                    (crossing[2]+1)%(2*len(self.crossings)),
                    (crossing[3]+1)%(2*len(self.crossings))
                ) 
                for crossing in b
            ]
            if set(a)==set(b): return True
        return False
    
    def isValid(self):
        h = 0
        for crossing in self.crossings:
            h ^= hash(crossing.up())
            h ^= hash(crossing.down())
            h ^= hash(crossing.left())
            h ^= hash(crossing.right())
        return h
    
    def __len__(self):
        return len(self.crossings)

def load(knotFile,knot):
    with open(f"./pd_codes/{knotFile}") as file:
        allFiles = json.loads(file.read())
        return Knot(allFiles[knot])
