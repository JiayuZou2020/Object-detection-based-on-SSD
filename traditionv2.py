import cv2
def is_inside(o, i):
    ox, oy, ow, oh = o
    ix, iy, iw, ih = i

    return ox > ix and oy > iy and ox + ow < ix + iw and oy + oh < iy + ih


def draw_person(img, person):
  x, y, w, h = person
  cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 255), 2)


img = cv2.imread("./data/car1.jpg")
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
found, w = hog.detectMultiScale(img, winStride=(8, 8), scale=1.05)


found_filtered = []
for ri, r in enumerate(found):
    for qi, q in enumerate(found):
        a = is_inside(r, q)
        if ri != qi and a:
            break
    else:
        found_filtered.append(r)

for person in found_filtered:
    draw_person(img, person)

cv2.imshow("detection", img)
cv2.waitKey(0)
cv2.destroyAllWindows()