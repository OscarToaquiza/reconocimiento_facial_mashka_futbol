from datetime import datetime

class Cronometer():
    
    @classmethod
    def obtener_tiempo_transcurrido_formateado(selft, hora_inicio):
        segundos_transcurridos = (datetime.now() - hora_inicio).total_seconds()
        return selft.segundos_a_segundos_minutos_y_horas(int(segundos_transcurridos))
        
    def segundos_a_segundos_minutos_y_horas(segundos):
        horas = int(segundos / 60 / 60)
        segundos -= horas*60*60
        minutos = int(segundos/60)
        segundos -= minutos*60
        return f"{horas:02d}:{minutos:02d}:{segundos:02d}"
