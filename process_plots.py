import os
from PIL import Image
import shutil

path_folder = "simulations"

#возможные значения CONFIDENCE_LEVEL_SHOP and CONFIDENCE_LEVEL_STOCK
values_confidence = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
# values_confidence = [0.5, 0.95]

# Возможные значения параметров
values = [1, 10, 100]

counter = 0
total_iterations = (len(values_confidence) ** 2) * (len(values))**4
print("Total iterations: ", total_iterations)

# Создаем новое изображение для сбора всех маленьких изображений
def create_concatenated_image(images):
    widths, heights = zip(*(i.size for i in images))

    total_width = sum(widths)//2
    max_height = sum(heights)//2

    new_im = Image.new('RGB', (total_width, max_height))

    x_offset = 0
    y_offset = 0
    for im in images:
        new_im.paste(im, (x_offset, y_offset))
        x_offset += im.size[0]
        if x_offset >= total_width:
            x_offset = 0
            y_offset += im.size[1]

    return new_im

# Итерация по всем комбинациям параметров
for products in values:
    for factories in values:
        for stock in values:
            for shop in values:
                image_list_obj = []
                image_list_loss = []
                print("Progress:", 100*counter/total_iterations,"%")
                for CONFIDENCE_LEVEL_SHOP in values_confidence:
                    for CONFIDENCE_LEVEL_STOCK in values_confidence:
                        path = f"{path_folder}/products_{products}/factories_{factories}/stock_{stock}/shop_{shop}"

                        img_obj = Image.open(os.path.join(path, f"objective_shop_{CONFIDENCE_LEVEL_SHOP}__stock_{CONFIDENCE_LEVEL_STOCK}.png"))  # Предполагаем, что изображения называются image.png
                        image_list_obj.append(img_obj)

                        img_loss = Image.open(os.path.join(path, f"loss_shop_{CONFIDENCE_LEVEL_SHOP}__stock_{CONFIDENCE_LEVEL_STOCK}.png"))
                        image_list_loss.append(img_loss)


                        path = f"{path_folder}/products_{products}/factories_{factories}/stock_{stock}/shop_{shop}"

                        path_for_images = path + "/images"
                        if not os.path.exists(path_for_images):
                            os.makedirs(path_for_images)

                        current_dst = os.path.join(path, f"objective_shop_{CONFIDENCE_LEVEL_SHOP}__stock_{CONFIDENCE_LEVEL_STOCK}.png")
                        new_file = os.path.join(path_for_images, f"objective_shop_{CONFIDENCE_LEVEL_SHOP}__stock_{CONFIDENCE_LEVEL_STOCK}.png")
                        shutil.move(current_dst, new_file)

                        current_dst = os.path.join(path, f"loss_shop_{CONFIDENCE_LEVEL_SHOP}__stock_{CONFIDENCE_LEVEL_STOCK}.png")
                        new_file = os.path.join(path_for_images, f"loss_shop_{CONFIDENCE_LEVEL_SHOP}__stock_{CONFIDENCE_LEVEL_STOCK}.png")
                        shutil.move(current_dst, new_file)

                        counter += 1
                
                # Создаем конкатенированное изображение из списка
                if image_list_obj:
                    final_image = create_concatenated_image(image_list_obj)
                    if final_image:
                        final_image.save(os.path.join(path, 'objective_image.png'))

                if image_list_loss:
                    final_image = create_concatenated_image(image_list_loss)
                    if final_image:
                        final_image.save(os.path.join(path, 'loss_image.png'))




                        