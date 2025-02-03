'''
    Второе задание тестирование.
    Автор: Лейман М.А.
'''

import unittest
from unittest.mock import MagicMock

# Класс CreditCard (мокируемый объект)
class CreditCard:
    def __init__(self, card_number, card_holder, expiry_date, cvv):
        self.card_number = card_number
        self.card_holder = card_holder
        self.expiry_date = expiry_date
        self.cvv = cvv

    def getCardNumber(self):
        return self.card_number

    def getCardHolder(self):
        return self.card_holder

    def getExpiryDate(self):
        return self.expiry_date

    def getCvv(self):
        return self.cvv

    def charge(self, amount: float):
        if amount > 1000:
            raise ValueError("Сумма списания превышает лимит")
        print(f"Списание ${amount} с карты{self.card_number}")
        return True


# Класс PaymentForm
class PaymentForm:
    def __init__(self, credit_card: CreditCard):
        self.credit_card = credit_card

    def pay(self, amount: float):
        if self.credit_card.charge(amount):
            return f"Платёж ${amount} прошёл успешно!"
        else:
            return "Платёж не прошёл"


class TestPaymentForm(unittest.TestCase):

    # Тесты для методов получения данных карты
    def test_get_card_number(self):
        mock_card = MagicMock(spec=CreditCard)
        mock_card.getCardNumber.return_value = "1234 5678 9012 3456"
        self.assertEqual(mock_card.getCardNumber(), "1234 5678 9012 3456")

    def test_get_card_holder(self):
        mock_card = MagicMock(spec=CreditCard)
        mock_card.getCardHolder.return_value = "Иван Иванов"
        self.assertEqual(mock_card.getCardHolder(), "Иван Иванов")

    def test_get_expiry_date(self):
        mock_card = MagicMock(spec=CreditCard)
        mock_card.getExpiryDate.return_value = "12/24"
        self.assertEqual(mock_card.getExpiryDate(), "12/24")

    def test_get_cvv(self):
        mock_card = MagicMock(spec=CreditCard)
        mock_card.getCvv.return_value = "123"
        self.assertEqual(mock_card.getCvv(), "123")

    # Тесты для метода charge
    def test_charge_successful(self):
        mock_card = MagicMock(spec=CreditCard)
        mock_card.charge.return_value = True

        result = mock_card.charge(100.0)
        self.assertTrue(result)

    def test_charge_exceeds_limit(self):
        mock_card = MagicMock(spec=CreditCard)
        mock_card.charge.side_effect = ValueError("Сумма списания превышает лимит")

        with self.assertRaises(ValueError):
            mock_card.charge(1500.00)

    # Позитивный тест для метода pay
    def test_successful_payment(self):
        mock_card = MagicMock(spec=CreditCard)
        mock_card.charge.return_value = True

        payment_form = PaymentForm(mock_card)
        amount = 100.00
        result = payment_form.pay(amount)

        mock_card.charge.assert_called_once_with(amount)
        self.assertEqual(result, f"Платёж ${amount} прошёл успешно!")

    # Отрицательный тест для метода pay
    def test_failed_payment(self):
        mock_card = MagicMock(spec=CreditCard)
        mock_card.charge.return_value = False

        payment_form = PaymentForm(mock_card)
        amount = 100.00
        result = payment_form.pay(amount)

        mock_card.charge.assert_called_once_with(amount)
        self.assertEqual(result, "Платёж не прошёл")

    # Тест для вызова charge в pay
    def test_charge_called_in_pay(self):
        mock_card = MagicMock(spec=CreditCard)
        mock_card.charge.return_value = True

        payment_form = PaymentForm(mock_card)
        payment_form.pay(200.00)

        # Проверяем, что charge был вызван с правильной суммой
        mock_card.charge.assert_called_once_with(200.00)


if __name__ == '__main__':
    unittest.main()



