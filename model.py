import matplotlib
matplotlib.use('TKAgg')

from threading import Thread, Condition, Lock, Timer
import time
import random
import Queue
from collections import OrderedDict
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation
import pylab
from pylab import *

realTimeCoefficient = 1    # how simulation time differs from real time 
                            # - 1 means simulation time is real time
                            # - otherwise - simulationTime = realTime/realTimeCoefficient
simulationTimeStep = 1 # milliseconds
networkDelay = 60
winsize = 15
rate = 30 # fps
gop = 30 # frames
testDuration = 20000 # milliseconds

simulationRunning = True
debugFlag = True

################################################
def logDebug(obj, string):
    global debugFlag
    global clock
    if debugFlag:
        print str(clock.getTime()) + '\t' + obj.__class__.__name__ + ':\t' + string

################################################
def log(obj, string):
    global clock
    print str(clock.getTime()) + '\t' + obj.__class__.__name__ + ':\t' + string

################################################
def stopSimulation():
    global simulationRunning 
    simulationRunning = False

################################################
class Clock(Thread):
    currentTime = 0
    step = 10
    coeff = 10
    lock = Lock()
    def __init__(self, step, coeff, stopTime = -1):
        Thread.__init__(self)
        self.step = step
        self.coeff = coeff
        self.currentTime = 0
        self.lock = Lock()
        self.stopTime = stopTime

    def run(self):
        while True:
            self.lock.acquire()
            self.currentTime += self.step
            self.lock.release()
            time.sleep(self.getRealTime(self.step)/1000.)
            if self.currentTime >= self.stopTime and self.stopTime > 0:
                stopSimulation()
                break
            # print "tick " + str(self.currentTime) + " " + str(self.getRealTime(self.step))

    def getTime(self):
        self.lock.acquire()
        time = self.currentTime
        self.lock.release()
        return time

    def getRealTime(self, simulationTime):
        return self.coeff*simulationTime

    def getSimulationTime(self, realTime):
        return realTime/self.coeff

################################################
class Packet():
    def __init__(self, seq, type):
        self.seq = seq
        self.timestamp = 0
        self.type = type # "data" or "interest"

    def __str__(self):
        return "<"+str(self.type)+"-"+str(self.seq)+">"

class FramePacket(Packet):
    def __init__(self, seq, dataType, frameType):
        Packet.__init__(self, seq, dataType)
        self.frameType = frameType

    def getKey(self):
        return 'k'+str(self.seq) if self.frameType == 'key' else 'd'+str(self.seq)

    def __str__(self):
        return "<"+str(self.type)+"-"+str(self.seq)+"-"+str(self.frameType)+">"

class Interest(FramePacket):
    def __init__(self, seq, frameType):
        FramePacket.__init__(self, seq, 'interest', frameType)

class Frame(FramePacket):
    def __init__(self, seq, frameType):
        FramePacket.__init__(self, seq, 'data', frameType)
        self.generationDelay = 0
        self.pairedNo = 0

################################################
class Network():
    def __init__(self, delay):
        self.delay = delay
        self.timers = {}
        self.cond = Condition()
        self.packetQueue = Queue.Queue()
        self.interestQueue = Queue.Queue()
        self.interestSubscribers = []
        self.dataSubscribers = []

    def send(self, packet):
        global clock
        self.cond.acquire()
        packet.timestamp = clock.getTime()
        if packet.type == 'data':
            # logDebug(self, 'sent data '+str(packet))
            self.packetQueue.put(packet)
        else:
            # logDebug(self, 'sent interest '+str(packet))
            self.interestQueue.put(packet)
        timer = Timer(clock.getRealTime(float(self.delay)/1000.), self.processPacket, args=[packet])
        self.timers[self.getStrId(packet)] = timer
        timer.start()
        self.cond.release()

    def subscribeForPackets(self, type, callback):
        if type == 'data':
            self.dataSubscribers.append(callback)
        else:
            self.interestSubscribers.append(callback)

    def processPacket(self, packet):
        # logDebug(self, 'process packet '+str(packet))
        self.cond.acquire()
        del self.timers[self.getStrId(packet)]
        self.cond.release()        
        # check subscribers
        subscribers = self.dataSubscribers if packet.type == 'data' else self.interestSubscribers
        for callback in subscribers:
            callback(packet)    

    def getStrId(self, packet):
        keyStr = 'd'+str(packet.seq) if packet.type == 'data' else 'i'+str(packet.seq)
        return keyStr+'d' if packet.frameType == 'delta' else keyStr+'k'

################################################
class Cache(Thread):
    def __init__(self):
        Thread.__init__(self)
        self.lock = Lock()
        self.interests = OrderedDict()
        self.delta = OrderedDict()
        self.key = OrderedDict()
        # self.data = OrderedDict()
        global network
        network.subscribeForPackets('interest', self.newInterest)

    def newInterest(self, interest):
        if interest:
            data = self.delta if interest.frameType == 'delta' else self.key
            if interest.seq == -1: # rightmost
                # logDebug(self, 'got rightmost')
                self.lock.acquire()
                last = data[data.keys()[len(data)-1]]
                self.lock.release()
                network.send(last)
            elif interest.getKey() in data.keys():
                # logDebug(self, 'interest hit cache '+str(interest))
                network.send(data[interest.getKey()])
                self.lock.acquire()
                del data[interest.getKey()]
                self.lock.release()
            else:
                # logDebug(self, 'no data for interest '+str(interest))
                self.lock.acquire()
                interest.timestamp = clock.getTime()
                self.interests[interest.getKey()] = interest
                self.lock.release()

    def add(self, frame):
        global network
        # logDebug(self, 'add data to cache ' + str(frame))
        self.lock.acquire()
        if frame.getKey() in self.interests.keys():
            # logDebug(self, 'got pending interest for data '+str(frame))
            frame.generationDelay = clock.getTime() - self.interests[frame.getKey()].timestamp
            network.send(frame)
            del self.interests[frame.getKey()]
        else:
            if frame.frameType == 'delta':
                self.delta[frame.getKey()] = frame
            else:
                self.key[frame.getKey()] = frame
        self.lock.release()

################################################
class ProducerThread(Thread):
    def __init__(self, rate, gop):
        Thread.__init__(self)
        self.interval = 1000./rate
        self.gop = gop
        self.seq = 0
        self.kseq = 0
        self.dseq = 0

    def run(self):
        global clock
        global cache
        global simulationRunning
        # logDebug(self, 'started')
        while True and simulationRunning:
            if self.seq % self.gop == 0:
                frame = Frame(self.kseq, "key")
                frame.pairedNo = self.dseq+1
                self.kseq = self.kseq + 1
            else:
                frame = Frame(self.dseq, "delta")
                frame.pairedNo = self.kseq
                self.dseq = self.dseq + 1
            self.seq = self.seq + 1
            # logDebug(self, 'adding to cache '+str(frame))
            cache.add(frame)
            time.sleep(clock.getRealTime(self.interval)/1000.)

################################################
class MeanEstimator():
    def __init__(self, sampleSize):
        self.valueMean = 0
        self.valueMeanSq = 0
        self.nValues = 0
        self.deviation = None
        self.sampleSize = sampleSize
        self.currentMean = 0

    def newValue(self, value):
        self.nValues = self.nValues + 1
        flush = False
        if self.nValues > 1:
            delta = value - self.valueMean
            self.valueMean = self.valueMean + float(delta)/float(self.nValues)
            self.valueMeanSq = self.valueMeanSq + delta*delta

            if self.sampleSize == 0 or self.nValues % self.sampleSize == 0:
                flush = True
                self.currentMean = self.valueMean
                if self.nValues >= 2:
                    variance = float(self.valueMeanSq)/float(self.nValues)
                    self.deviation = math.sqrt(variance)
                if self.sampleSize == 0:
                    self.nValues = 0
        else:
            flush = True

        if flush:
            # self.currentMean = value
            self.valueMean = value
            self.valueMeanSq = 0

    def getCurrentMeanValue(self):
        return self.currentMean

    def getCurrentDeviationValue(self):
        return self.deviation

class MovingAverage():
    def __init__(self, sampleSize):
        self.sampleSize = sampleSize if sampleSize > 1 else 2
        self.nValues = 0
        self.accumulatedSum = 0
        self.currentAverage = 0
        self.sample = [0.]*sampleSize

    def newValue(self, value):
        self.sample[self.nValues%self.sampleSize] = value
        self.nValues = self.nValues + 1
        if self.nValues >= self.sampleSize:
            self.currentAverage = float(self.accumulatedSum + value) / float(self.sampleSize)
            self.accumulatedSum = self.accumulatedSum + value - self.sample[self.nValues%self.sampleSize]
        else:
            self.accumulatedSum = self.accumulatedSum + value

    def getCurrentValue(self):
        return self.currentAverage

class InclineEstimator():
    def __init__(self, sampleSize):
        self.sampleSize = sampleSize
        self.nValues = 0
        self.avgEstimator = MovingAverage(sampleSize)
        self.lastValue = None

    def newValue(self, value):
        if self.nValues == 0:
            self.lastValue = float(value)
        else:
            incline = float(value) - self.lastValue
            self.avgEstimator.newValue(incline)
            self.lastValue = value
        self.nValues = self.nValues + 1

    def getCurrentValue(self):
        return self.avgEstimator.getCurrentValue()

class  LowPassFilter(object):
    """docstring for  LowPassFilter"""
    def __init__(self, alpha):
        super(LowPassFilter, self).__init__()
        self.alpha = alpha
        self.filteredValue = 0

    def newValue(self, value):
        self.filteredValue = self.filteredValue + self.alpha*float(value - self.filteredValue)
 
    def getCurrentValue(self):
        return self.filteredValue       

################################################
class ExhaustionEstimator():
    rateSimilarityLevel = 0.7
    def __init__(self, sampleSize, threshold, minOccurences):
        self.minStableOccurences = minOccurences
        self.sampleSize = sampleSize
        self.stabilityThreshold = threshold
        self.flush()

    def newValue(self, value):
        # global clock
        # t = clock.getTime()
        # interarrival = t - self.lastDataTimestamp if self.lastDataTimestamp != 0 else 0
        # self.lastDataTimestamp = t
        self.meanEstimator.newValue(value)
        mean = self.meanEstimator.getCurrentMeanValue()

        global rate
        deviationPercentage = -1
        if not self.meanEstimator.getCurrentDeviationValue() == None and not self.meanEstimator.getCurrentMeanValue() == 0:
            deviationPercentage = self.meanEstimator.getCurrentDeviationValue() / self.meanEstimator.getCurrentMeanValue()
            targetDelay = 1000./float(rate)
            if deviationPercentage <= self.stabilityThreshold and abs(mean-targetDelay)/targetDelay <= 1.-ExhaustionEstimator.rateSimilarityLevel:
                self.nStableOccurrences = self.nStableOccurrences + 1
            else:
                self.stable = False
                self.nStableOccurrences = 0

        if self.nStableOccurrences >= self.minStableOccurences:
            self.stable = True

    def getMeanValue(self):
        # return self.lowPassFilter.getCurrentValue()
        return self.meanEstimator.getCurrentMeanValue()

    def isStable(self):
        return self.stable

    def flush(self):
        self.meanEstimator = MeanEstimator(self.sampleSize)
        self.nStableOccurrences = 0
        self.stable = False

class RttChangeEstimator():
    rateSimilarityLevel = 0.7
    def __init__(self, sampleSize, threshold, minOccurences):
        self.minStableOccurences = minOccurences
        self.sampleSize = sampleSize
        self.stabilityThreshold = threshold
        self.flush()

    def newValue(self, value):
        self.meanEstimator.newValue(value)
        mean = self.meanEstimator.getCurrentMeanValue()

        deviationPercentage = -1
        if not self.meanEstimator.getCurrentDeviationValue() == None and not self.meanEstimator.getCurrentMeanValue() == 0:
            deviationPercentage = self.meanEstimator.getCurrentDeviationValue() / self.meanEstimator.getCurrentMeanValue()
            if deviationPercentage <= self.stabilityThreshold:
                self.nStableOccurrences = self.nStableOccurrences + 1
            else:
                self.stable = False
                self.nChanges = 0
                self.nMinorConsecutiveChanges = 0
                self.nStableOccurrences = 0

        if self.nStableOccurrences >= self.minStableOccurences:
            self.stable = True

        if self.stable:
            changePercentage = abs(value - mean)/mean
            if changePercentage > 0.08:
                if changePercentage < 0.2:
                    self.nMinorConsecutiveChanges = self.nMinorConsecutiveChanges + 1
                else:
                    self.nChanges = self.nChanges + 1
            else:
                self.nMinorConsecutiveChanges = 0

    def getMeanValue(self):
        # return self.lowPassFilter.getCurrentValue()
        return self.meanEstimator.getCurrentMeanValue()

    def isStable(self):
        return self.stable

    def hasChange(self):
        result = False
        if self.nChanges > self.lastCheckedChangeNumber:
            result = True
        self.lastCheckedChangeNumber = self.nChanges
        return result

    def flush(self):
        self.meanEstimator = MeanEstimator(self.sampleSize)
        self.nStableOccurrences = 0
        self.stable = False
        self.nChanges = 0
        self.nMinorConsecutiveChanges = 0
        self.lastCheckedChangeNumber = 0

################################################
class ConsumerThread(Thread):
    def __init__(self, winsize):
        Thread.__init__(self)
        self.rtt = {}
        self.defaultWindow = winsize
        self.window = winsize
        self.lock = Lock()
        self.cond = Condition()
        self.seq = -1
        self.dseq = 0
        self.kseq = 0
        self.needKey = False
        self.lastDataTimestamp = 0
        self.exhaustionEstimator = ExhaustionEstimator(1, 0.08, 3)
        self.rttStabilityEstimator = RttChangeEstimator(1, 0.1, 3)
        self.startTime = 0
        self.mode = 'chase' # 'adjust', 'fetch'
        self.adjustFirstInterest = 0
        self.adjustLastInterest = 0
        self.needAdjustment = True
        self.decreaseCount = 0
        self.stopDecrease = False
        self.adjustIteration = 0
        global gop
        self.producerGop = gop
        self.lastStableWindowSize = 0
        self.darrLP = 0
        self.alpha = 0.05

        self.waitForChange = False
        self.waitForStability = False
        self.timerStartTime = 0
        self.minWindow = 0

    def run(self):
        global network
        global simulationRunning
        # logDebug(self, 'started')
        # self.startTime = clock.getTime()
        self.timerStartTime = clock.getTime()
        network.subscribeForPackets('data', self.newData)
        # chasing - issue rightmost
        interest = Interest(-1, 'key')
        # logDebug(self, 'interest '+str(interest))
        network.send(interest)
        self.cond.acquire()
        # logDebug(self, 'waiting for rightmost data...')
        self.rtt[interest.seq] = clock.getTime()
        self.cond.wait()
        self.cond.release()

        while True and simulationRunning:
            self.cond.acquire()
            interest = None
            if self.needKey:
                interest = Interest(self.kseq, 'key')
                self.kseq = self.kseq + 1
                self.needKey = False
            elif self.window > 0:
                interest = Interest(self.dseq, 'delta')
                self.dseq = self.dseq + 1

            if interest:
                # logDebug(self, 'interest '+str(interest))
                self.window = self.window - 1
                self.rtt[interest.seq] = clock.getTime()
                network.send(interest)
            else:
                self.cond.wait()
            self.cond.release()

    def newData(self, packet):
        global clock
        global plotter
        if not packet.seq in self.rtt.keys():
            self.rtt[packet.seq] = clock.getTime() - self.rtt[-1]
            self.kseq = packet.seq+1
            self.dseq = packet.pairedNo
        else:
            self.rtt[packet.seq] = clock.getTime() - self.rtt[packet.seq]
        
        self.cond.acquire()
        t = clock.getTime()

        if packet.frameType == 'key':
            self.needKey = True

        self.window = self.window + 1
        interarrival = t - self.lastDataTimestamp if self.lastDataTimestamp != 0 else 0
        self.exhaustionEstimator.newValue(interarrival)
        self.darrLP = self.darrLP + self.alpha*float(interarrival - self.darrLP)
        self.lastDataTimestamp = t

        needIncrease = False
        needDecrease = False
        timeout = 1500

        if self.mode == 'chase' and t-self.timerStartTime > timeout and not self.exhaustionEstimator.isStable():
            print "increase after chase"
            needIncrease = True
            self.timerStartTime = t
            self.waitForStability = True

        if not self.exhaustionEstimator.isStable():
            self.rttStabilityEstimator.flush()
        else:
            if self.mode == 'chase':
                self.mode = 'adjust'
            if not packet.frameType == 'key':
                self.rttStabilityEstimator.newValue(self.rtt[packet.seq])

        if self.mode == 'adjust':
            if self.rttStabilityEstimator.isStable():
                if self.waitForChange:
                    if self.rttStabilityEstimator.hasChange():
                        print "has change"
                        self.rttStabilityEstimator.flush()
                        self.waitForChange = False
                        self.waitForStability = True
                        self.timerStartTime = t
                    elif t-self.timerStartTime > timeout:
                        print "timeout waiting for change"
                        needDecrease = True
                        self.timerStartTime = t
                else:
                    print "decrease"
                    needDecrease = True
                    self.waitForChange = True
                    self.waitForStability = False
                    self.timerStartTime = t
            elif t-self.timerStartTime > timeout:
                print "increase"
                self.minWindow = self.defaultWindow
                needIncrease = True
                self.timerStartTime = t
                self.waitForStability = True
                self.waitForChange = False

        if needIncrease or needDecrease:
            # change window size
            if needIncrease:
                if self.decreaseCount == 0:
                    delta = int(math.ceil(1*float(self.defaultWindow)))
                    self.window = self.window + delta
                    self.defaultWindow = self.defaultWindow+delta
                else: # grow slower
                    delta = int(math.ceil(0.25*float(self.defaultWindow)))
                    self.window = self.window + delta
                    self.defaultWindow = self.defaultWindow + delta
            elif needDecrease:
                delta = int(math.ceil(0.25*float(self.defaultWindow)))
                if self.minWindow >= self.defaultWindow - delta:
                    if self.minWindow+1 >= self.defaultWindow:
                        print 'trying to decrease lower than minWindow. stop adjustment'
                        self.stopDecrease = True
                        self.mode = 'fetch'
                
                if not self.mode == 'fetch':
                    if self.defaultWindow - delta > 0:
                        self.defaultWindow = self.defaultWindow - delta
                        self.window = self.window - delta
            if self.mode == 'adjust':
                print 'ADJUST: new window '+ str(self.window) + ' default window ' + str(self.defaultWindow)

        stableFlag = 'stable' if self.exhaustionEstimator.isStable() else 'unstable'
        rttStableFlag = 'rtt stable' if self.rttStabilityEstimator.isStable() else 'rtt unstable'
        change = 'change detected' if self.rttStabilityEstimator.hasChange() else 'no change'
        logDebug(self, str(packet)+'\trtt\t'+str(self.rtt[packet.seq]) + '\twin\t'+str(self.window)+\
            '\tdarr\t'+str(interarrival) + '\tincl m \t'+\
            '\tincl \t'+'\t\t'+stableFlag+\
            '\t'+rttStableFlag+'\t'+str(packet.generationDelay)+'\t)' )

        self.cond.notify()
        self.cond.release()


################################################
class Plotter():
    def __init__(self):
        self.lock = Lock()
        self.xAchse=pylab.arange(0,100,1)
        self.yAchse=pylab.array([0]*100)
        self.fig = pylab.figure(1)
        self.ax = self.fig.add_subplot(111)
        self.ax.grid(True)
        self.ax.set_xlabel("Time")
        self.ax.axis([0,100,-1.5,1.5])
        self.manager = pylab.get_current_fig_manager()
        self.lines = {}
        # pyplot.legend()

    def start(self):
        pylab.show(block=False)
        self.setTimer()

    def drawLines(self):
        global simulationRunning
        currentXAxis = None
        self.lock.acquire()
        xMax = 0
        xMin = 0
        yMax = 0
        yMin = 0
        for lineName in self.lines.keys():
            line = self.lines[lineName]['line']
            values = self.lines[lineName]['values']
            if currentXAxis == None:
                currentXAxis=pylab.arange(len(values)-100,len(values),1)
            line[0].set_data(currentXAxis,pylab.array(values[-100:]))
            # if currentXAxis.min() < xMin:
            #     xMin = currentXAxis.min()
            # if currentXAxis.max() > xMax:
            #     xMax = currentXAxis.max()
            if min(values) < yMin:
                yMin = min(values)
            if max(values) > yMax:
                yMax = max(values)
            # self.ax.axis([xMin,xMax,yMin*1.1,yMax*1.2])
        self.ax.axis([currentXAxis.min(),currentXAxis.max(),yMin*1.1,yMax*1.2])
        self.manager.canvas.draw()
        self.lock.release()
        if simulationRunning:
            self.setTimer()

    def setTimer(self):
        self.timer = Timer(clock.getRealTime(float(simulationTimeStep)/1000.), self.drawLines, args=[])
        self.timer.start()

    def addLine(self, lineName, marker):
        newLine = self.ax.plot(self.xAchse, self.yAchse, marker, label=lineName)
        self.lock.acquire()
        self.lines[lineName] = {'line':newLine, 'values':[0 for x in range(100)]}
        self.lock.release()

    def newData(self, lineName, x, y):
        self.lock.acquire()
        self.lines[lineName]['values'].append(y)
        self.lock.release()


################################################
print "sim step: " + str(simulationTimeStep) + \
        " rt coeff: " + str(realTimeCoefficient) + \
        " winsize: " + str(winsize) + \
        " network delay: " + str(networkDelay) + \
        " rate: " + str(rate) + \
        " gop: " + str(gop) + \
        " simulation duration: " + str(testDuration)
# print headers
print "\t\t\t\tRTT\t\t\t\tDarr"

clock = Clock(simulationTimeStep, realTimeCoefficient, testDuration)
clock.start()

network = Network(networkDelay)
cache = Cache()

producer = ProducerThread(rate,gop)
producer.start()
time.sleep(clock.getRealTime(0.3))

consumer = ConsumerThread(winsize)
consumer.start()

# plotter = Plotter()
# plotter.addLine('int','*')
# plotter.addLine('data','^')
# plotter.addLine('RTT','-')
# plotter.addLine('Darr', '-')
# plotter.addLine('win', '-')
# plotter.addLine('Key', '^')
# plotter.start()
