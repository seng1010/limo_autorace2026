"""M8 observers reuse already decoded images/paths; no second lane pipeline."""
import math
from typing import Optional
import cv2
import numpy as np
from jetracer_ros2.stop_line_detector import StopLineDetection


def _entry_contours(image_bgr, roi_top_ratio):
    height, width = image_bgr.shape[:2]
    top = int(height * roi_top_ratio)
    roi = image_bgr[top:]
    if roi.size == 0:
        return [], None, top, height, width
    hls = cv2.cvtColor(roi, cv2.COLOR_BGR2HLS)
    mask = cv2.inRange(hls, (0, 145, 0), (180, 255, 125))
    kernel = np.ones((2, max(3, round(.02*width))), np.uint8)
    opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    contours, _ = cv2.findContours(opened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours, hls, top, height, width


def detect_m8_stop_line(image_bgr: np.ndarray, roi_top_ratio: float, *,
                        allow_oblique: bool = True) -> Optional[StopLineDetection]:
    """Find the slightly wavy, shaded white tape across the M8 lane mouth."""
    contours, _, top, height, width = _entry_contours(image_bgr, roi_top_ratio)
    return _detect_entry(contours, top, height, width, allow_oblique)


def _detect_entry(contours, top, height, width, allow_oblique):
    best = None
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        # A real entry line spans the lane, including the camera center. Reject
        # wall/floor bands, clipped blobs, side markings and long thin glare.
        if not (w >= .38*width and 4 <= h <= .11*height and w >= 5*h and
                x <= .47*width and x+w >= .53*width and
                .065*height <= y+h < height-top-5 and cv2.contourArea(contour) >= .26*w*h):
            continue
        if best is None or y+h > best[1]+best[3]:
            best = (x, y, w, h)
    if best is None:
        return _detect_oblique_entry(contours, top, height, width) if allow_oblique else None
    x, y, w, h = best
    return StopLineDetection((top+y+h)/height, (x, top+y, w, h))


def _detect_oblique_entry(contours, top, height, width):
    """Fit the tape along its axis; calibrate the edge at the camera center."""
    best = None
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        # A tape just grazing the center during the curve is too oblique
        # in position to use as a front-axle anchor. Require support on
        # both sides of the center, though either far end may be offscreen.
        if not (w >= .38*width and x <= .365*width and x+w >= .635*width
                and y+h < height-top-5):
            continue
        (cx, cy), (length, thick), angle = cv2.minAreaRect(contour)
        if length < thick:
            length, thick, angle = thick, length, angle+90
        angle = (angle+90)%180-90
        if not (4. <= abs(angle) <= 25. and 4. <= thick <= .09*height
                and length >= 5*thick and cv2.contourArea(contour) >= .35*length*thick):
            continue
        angle = math.radians(angle)
        bottom = cy + (width/2-cx)*math.tan(angle) + thick/(2*math.cos(angle))
        if not .065*height <= bottom < height-top-5:
            continue
        if best is None or bottom > best[0]:
            best = bottom, (x, top+y, w, h)
    return StopLineDetection((top+best[0])/height, best[1]) if best else None


def _detect_partial_entry(contours, hls, top, height, width):
    """A nearby tape cue, NOT permission to drive or a full-width detection.

    Yellow contact distinguishes a transverse entry-tape end from the white
    mat edge. Limit extrapolation to a quarter image width. The projected row
    is only an approximate entrance anchor; a fresh safe cone command must
    independently authorize handover in M2.
    """
    if hls is None:
        return None
    yellow = cv2.inRange(hls, (12, 50, 80), (40, 255, 255))
    radius = max(3, round(.014*width))
    nearby_yellow = cv2.dilate(yellow, np.ones((2*radius+1, 2*radius+1), np.uint8))
    best = None
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        gap = max(x-width/2, width/2-(x+w), 0.)
        if w < .22*width or gap > .25*width or y < 2:
            continue
        (cx, cy), (length, thick), angle = cv2.minAreaRect(contour)
        if length < thick:
            length, thick, angle = thick, length, angle+90
        angle = (angle+90)%180-90
        clipped = y+h >= height-top-5
        if not (abs(angle) <= 25. and 4. <= thick <= (.12 if clipped else .10)*height
                and length >= (3.5 if clipped else 5.)*thick
                and cv2.contourArea(contour) >= .35*length*thick):
            continue
        if clipped and not (x <= width/2 <= x+w and w >= .25*width):
            continue
        xy = contour[:, 0]
        if np.count_nonzero(nearby_yellow[xy[:, 1], xy[:, 0]]) < 2:
            continue
        radians = math.radians(angle)
        row = (top+cy+(width/2-cx)*math.tan(radians)+thick/(2*math.cos(radians)))/height
        # Stay within the useful near-ground image band, including when the
        # far end is clipped. Never project below the image or to the horizon.
        if not .65 <= row <= .98:
            continue
        if best is None or row > best.row_ratio:
            best = StopLineDetection(row, (x, top+y, w, h))
    return best


def detect_m8_entry_candidates(image, roi_top_ratio):
    """Share one color transform/morphology pass for strong and partial cues."""
    contours, hls, top, height, width = _entry_contours(image, roi_top_ratio)
    full = _detect_entry(contours, top, height, width, True)
    partial = None if full else _detect_partial_entry(contours, hls, top, height, width)
    return full, partial


def detect_m8_exit_stop_line(image: np.ndarray) -> Optional[StopLineDetection]:
    """Find oblique/partially visible exit tape; never used for M8 entry or M1/M3."""
    if image.size == 0:
        return None
    height, width = image.shape[:2]
    top = int(height * .49)
    hls = cv2.cvtColor(image[top:], cv2.COLOR_BGR2HLS)
    mask = cv2.inRange(hls, (0,145,0), (180,255,125))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3,7), np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best = None
    for contour in contours:
        (cx, cy), (length, thick), angle = cv2.minAreaRect(contour)
        if length < thick:
            length, thick, angle = thick, length, angle+90
        angle = (angle+90)%180-90
        if not (.15*width <= length and 2 <= thick <= .095*height
                and length >= 5*thick and abs(angle) <= 25):
            continue
        area = cv2.contourArea(contour)
        if area < .19*length*thick:
            continue
        x,y,w,h = cv2.boundingRect(contour)
        if y < 2 or y+h >= len(mask)-3:
            continue
        # Project only a limited lateral gap to the middle; further out the
        # image-row calibration does not represent the approaching axle.
        lateral_allowance = .33
        if not x-lateral_allowance*width <= width/2 <= x+w+lateral_allowance*width:
            continue
        center_y = top + cy + (width/2-cx)*math.tan(math.radians(angle))
        bottom = center_y+thick/2
        if not .51*height <= bottom <= .98*height:
            continue
        score = length + .4*min(100.,area/thick) - .3*abs(cx-width/2)
        if best is None or score > best[0]:
            best = score, StopLineDetection(bottom/height,(x,top+y,w,h))
    return best[1] if best else None


class M8LaneObserver:
    def __init__(self,node,forward_offset):
        from geometry_msgs.msg import PolygonStamped, TwistStamped
        from jetracer_ros2.m8_entry import OBSERVED_LANE_TOPIC,M8_LANE_PATH_TOPIC,M8_LANE_HINT_TOPIC
        self.node,self.forward_offset=node,forward_offset
        self.command=node.create_publisher(TwistStamped,OBSERVED_LANE_TOPIC,1)
        self.path=node.create_publisher(PolygonStamped,M8_LANE_PATH_TOPIC,1)
        self.hint=node.create_publisher(TwistStamped,M8_LANE_HINT_TOPIC,1)

    def update(self,stamp,result,cmd,resetting=False):
        from geometry_msgs.msg import Point32,PolygonStamped,TwistStamped
        from jetracer_ros2.m8_entry import MIN_M8_SPEED,M8_LANE_PATH_FRAME
        from jetracer_ros2.utils.ackermann_utils import inner_angle_to_omega
        if not any(p.get_subscription_count() for p in (self.command,self.path,self.hint)):
            return
        observed,path,hint=TwistStamped(),PolygonStamped(),TwistStamped()
        for message in (observed,path,hint):
            message.header.stamp=stamp
            message.header.frame_id=M8_LANE_PATH_FRAME
        if result is not None and not resetting:
            seen=result['left'] is not None or result['right'] is not None
            if cmd[2]=='TRACK' and seen:
                observed.twist.linear.x=float(cmd[0])
                observed.twist.angular.z=float(inner_angle_to_omega(cmd[1],cmd[0]))
            if seen and result['valid'] and result['center'] is not None:
                right,forward=result['center']
                path.polygon.points=[Point32(x=float(x+self.forward_offset),y=float(-y))
                                     for x,y in zip(forward,right)]
                hint.twist.linear.x=MIN_M8_SPEED
                hint.twist.angular.z=float(inner_angle_to_omega(result['raw_angle'],MIN_M8_SPEED))
        self.command.publish(observed)
        self.path.publish(path)
        self.hint.publish(hint)


class M8StopLineObserver:
    def __init__(self,node,model,config):
        from geometry_msgs.msg import PointStamped
        from std_msgs.msg import Float32,String
        from rclpy.qos import QoSProfile,DurabilityPolicy
        from jetracer_ros2.m8_entry import M8_EXIT_LINE_TOPIC,M8_PHASE_TOPIC,M8_PARTIAL_ENTRY_TOPIC
        self.node,self.model,self.config=node,model,config
        self.phase='off'
        self.entry=node.create_publisher(Float32,'mission/stop_line_visual_distance',1)
        self.exit=node.create_publisher(PointStamped,M8_EXIT_LINE_TOPIC,1)
        self.partial=node.create_publisher(PointStamped,M8_PARTIAL_ENTRY_TOPIC,1)
        node.create_subscription(String,M8_PHASE_TOPIC,self._phase,
                                 QoSProfile(depth=1,durability=DurabilityPolicy.TRANSIENT_LOCAL))

    def _phase(self,msg):
        self.phase=msg.data

    def update(self,image,header):
        from geometry_msgs.msg import PointStamped
        from std_msgs.msg import Float32
        stamp=header.stamp.sec*10**9+header.stamp.nanosec
        age=(self.node.get_clock().now().nanoseconds-stamp)/1e9
        if self.phase not in ('watch_entry','preflight','approach','cone','exit_prepare','exit_lane') or stamp<=0 or not 0<=age<=.25:
            return
        # During cone ownership keep the original entry-anchor fallback;
        # the relaxed geometry is for M2 approach only.
        if self.phase == 'cone':
            entry=detect_m8_stop_line(image,self.config.roi_top_ratio,allow_oblique=False)
            self.entry.publish(Float32(data=self.model.front_distance(entry.row_ratio) if entry else -1.))
        elif self.phase in ('watch_entry','preflight','approach'):
            entry,partial=detect_m8_entry_candidates(image,self.config.roi_top_ratio)
            cue=PointStamped()
            cue.header.stamp=header.stamp
            cue.header.frame_id='front_axle'
            cue.point.x=(self.model.front_distance(partial.row_ratio)
                         if partial and self.model.calibrated else -1.)
            self.partial.publish(cue)
            self.entry.publish(Float32(data=self.model.front_distance(entry.row_ratio) if entry else -1.))
        if self.phase in ('cone','exit_prepare','exit_lane'):
            detected=detect_m8_exit_stop_line(image)
            if detected:
                msg=PointStamped()
                msg.header.stamp=header.stamp
                msg.header.frame_id='front_axle'
                msg.point.x=self.model.front_distance(detected.row_ratio)
                self.exit.publish(msg)
