import cv2, os
import sys
from .utils import ensure_dir, get_face_detector, detect_faces

def capture_for_person(name, out_dir='dataset', samples=50):
    ensure_dir(out_dir)
    person_dir = os.path.join(out_dir, name)
    ensure_dir(person_dir)
    cap = cv2.VideoCapture(0)
    detector = get_face_detector()
    count = 0
    print('Press "q" to quit early.')
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        faces = detect_faces(frame, detector)
        for (x,y,w,h, face) in faces:
            face_resized = cv2.resize(face, (160,160))
            path = os.path.join(person_dir, f'{count:03d}.png')
            cv2.imwrite(path, face_resized)
            count += 1
            cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 2)
            cv2.putText(frame, f'{count}/{samples}', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
        cv2.imshow('Capture - Press q to stop', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        if count >= samples:
            break
    cap.release()
    cv2.destroyAllWindows()
    print(f'Captured {count} images for {name}.')

if __name__ == '__main__':
    if len(sys.argv) >= 2:
        name = sys.argv[1]
        samples = int(sys.argv[2]) if len(sys.argv) >= 3 and sys.argv[2].isdigit() else 50
    else:
        name = input('Enter person name or ID (no spaces): ').strip()
        samples = input('How many samples to capture per person? [default 50]: ').strip()
        samples = int(samples) if samples.isdigit() else 50
    capture_for_person(name, samples=samples)
