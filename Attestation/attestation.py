
import numpy as np

def visualize_image_and_mask(image, mask, class_palette):
    mask_np = np.array(mask)
    
    # Создаём копию маски для отображения
    mask_vis = mask_np.copy()
    
    # Заменяем 255 (ignore) на 0 или другой индекс, например фон
    mask_vis[mask_vis == 255] = 19
    
    # Применяем палитру
    color_mask = class_palette[mask_vis]
    
    # Отображение
    import matplotlib.pyplot as plt
    
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    axs[0].imshow(image)
    axs[0].set_title('Image')
    axs[0].axis('off')
    
    axs[1].imshow(color_mask)
    axs[1].set_title('Mask')
    axs[1].axis('off')
    
    plt.show()








